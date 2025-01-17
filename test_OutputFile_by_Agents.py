#!pip install langgraph langchain_openai langchain_core

import csv
import torch
import os
import re
import time
from openai import AzureOpenAI
import openai

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.base import Runnable

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_google_genai import ChatGoogleGenerativeAI

#=================Parameter===========================================================================
# (1) ここで宣言しているパラメータ（グローバル変数）は、ユーザーが変更しそうなものだけに絞っています。
# これ以外の変数はメイン関数または必要な関数の内部ローカル変数として定義し、グローバルスコープを汚さない方針とします。

# 採点者モデルの選択
# "OpenAI_Base"では、gpt-4o系統もしくは、deepseekのみ実装済み
#Evaluation = "OpenAI_Base"
#Evaluation_model = "deepseek-chat"
#Evaluation = "OpenAI_Base"
#Evaluation_model = "gpt-4o-mini"
Evaluation = "Azure"
Evaluation_model = "gpt-4o"
#Evaluation = "Google"
#Evaluation_model = "gemini-2.0-flash-exp"
#Evaluation = "HuggingFace"
#Evaluation_model = "meta-llama/Llama-3.3-70B-Instruct"

# 評価対象のファイル群を選定する
output_txt = "./outputs/Google-gemini-2.0-flash-exp/output-Google-gemini-2.0-flash-exp.txt"

# 何問目から再開するか (1問目から始める場合は1)
resume_question_index = 1

# Huggingfaceにて、アクセス権限がないと取得できないモデルを利用するかどうかのフラグ
HuggingFace_access = True

# 採点側温度の設定 (評価に使うので、基本的に0.001でいい)
Evaluation_temperature = 0.001

# 採点側top_pの設定 (評価に使うので、基本的に0.001でいい)
Evaluation_top_p = 0.001

# AWSなどでキャッシュディレクトリを利用する場合はここに指定する。使わない場合はNone
efs_cache_dir = None

# CSVファイルのパス
csv_file = './inputs/test.csv'
#=======================================================================================================


def sanitize_filename(filename: str) -> str:
    """
    ファイル名に使用できない文字を安全に置き換える関数。

    Args:
        filename (str): 元のファイル名。

    Returns:
        str: 安全なファイル名。
    """
    # 不適切な文字をハイフンまたはアンダースコアに置き換える
    sanitized = re.sub(r'[\/:*?"<>|]', '-', filename)
    return sanitized


def get_file_paths(
    output_txt_path: str, 
    evaluation_name: str, 
    evaluation_model_name: str
) -> dict:
    """
    出力ファイルのパスを生成する。ユーザーが指定した output_txt をもとに、
    result_file, critc_file, score_file, markdown_output などのファイルパスを作成。

    Args:
        output_txt_path (str): 出力ファイル(txt)へのパス
        evaluation_name (str): 評価モデル名
        evaluation_model_name (str): 評価モデルの詳細名

    Returns:
        dict: 各ファイルのパスを格納した辞書
    """
    # ベースファイル名を取得 (拡張子を含む)
    base_name = os.path.basename(output_txt_path)
    # "output-" の後と ".txt" の前を取得
    safe_target_model = base_name.replace("output-", "").replace(".txt", "")
    safe_evaluation_model = sanitize_filename(evaluation_model_name)

    output_dir = f"./outputs/{safe_target_model}"
    os.makedirs(output_dir, exist_ok=True)

    return {
        "result_file": f"{output_dir}/result-{safe_target_model}_by_{safe_evaluation_model}.txt",
        "critc_file": f"{output_dir}/cretical-{safe_target_model}_by_{safe_evaluation_model}.txt",
        "score_file": f"{output_dir}/score-{safe_target_model}_by_{safe_evaluation_model}.txt",
        "markdown_output": f"{output_dir}/Elyza-{safe_target_model}_by_{safe_evaluation_model}.md",
        "safe_target_model": safe_target_model,
        "safe_evaluation_model": safe_evaluation_model,
    }


def initialize_evaluation_model(
    evaluation_name: str,
    evaluation_model_name: str,
    temperature: float,
    top_p: float
):
    """
    採点用のモデルを初期化して返す関数。
    (Evaluation, Evaluation_model, Evaluation_temperature, Evaluation_top_p 等を利用)

    Args:
        evaluation_name (str): 評価モデル名 ("Azure", "Google", "HuggingFace" 等)
        evaluation_model_name (str): 評価モデルの詳細名
        temperature (float): 温度
        top_p (float): top_p

    Returns:
        model: 初期化された評価モデル
    """
    if evaluation_name == "Azure":
        # 環境変数を登録するもしくは、下書きでも良い
        # os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
        # os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxxx.openai.azure.com"
        # os.environ["AZURE_OPENAI_API_KEY"] = "AtNixxxxxxxxxxxxxxxxxxxxx"
        os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "")
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")

        model_obj = AzureChatOpenAI(
            azure_deployment=evaluation_model_name,
            temperature=temperature,
        )

    elif evaluation_name == "Google":
        # 環境変数を登録するもしくは、直書きでも良い
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

        model_obj = ChatGoogleGenerativeAI(
            model=evaluation_model_name,
            temperature=temperature
            # top_pは引数で与えていないが必要であれば追記
        )

    elif evaluation_name == "HuggingFace":
        do_sample = (temperature > 0)
        huggingface_model = AutoModelForCausalLM.from_pretrained(
            evaluation_model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=efs_cache_dir,
            force_download=False,
            trust_remote_code=True
        )

        dtypes_llama = {param.dtype for param in huggingface_model.parameters()}
        print(f"モデルで使用されているデータ型: {dtypes_llama}")

        tokenizer = AutoTokenizer.from_pretrained(
            evaluation_model_name,
            cache_dir=efs_cache_dir,
            force_download=False,
            trust_remote_code=True,
            use_fast=True
        )

        # LLaMa系の場合にテンプレートを読み込む
        if (tokenizer.chat_template is None) and ("llama" in evaluation_model_name.lower()):
            with open("./inputs/llama_chat_template", 'r', encoding='utf-8') as file:
                template = file.read()
            tokenizer.chat_template = template

        pipe = pipeline(
            "text-generation",
            model=huggingface_model,
            tokenizer=tokenizer,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=1024
        )
        pipe = HuggingFacePipeline(pipeline=pipe)
        model_obj = ChatHuggingFace(llm=pipe)

    elif evaluation_name == "OpenAI_Base":
        API_KEY = None
        ENDPOINT = None
        if "gpt" in evaluation_model_name:
            API_KEY = os.getenv("OPENAI_API_KEY", "")
            ENDPOINT = "https://api.openai.com/v1"
        elif evaluation_model_name == "deepseek-chat":
            API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
            ENDPOINT = "https://api.deepseek.com"
        else:
            raise ValueError("モデルが不正です。")

        model_obj = ChatOpenAI(
            model=evaluation_model_name,
            openai_api_key=API_KEY,
            openai_api_base=ENDPOINT,
            max_tokens=4096,
            temperature=temperature,
            top_p=top_p,
            model_kwargs={"stream": False}
        )

    else:
        model_obj = None
        print("モデルが選択されていません。")
        exit()

    return model_obj


def read_output_file(output_txt_path: str) -> list:
    """
    被評価モデルが出力したテキストファイル(output_txt)を読み込み、
    各問題(Question)と回答(Answer)を正規表現で分割して返す関数。

    Args:
        output_txt_path (str): 被評価モデルのテキスト出力ファイルのパス

    Returns:
        list: 
            [
              {
                "Question Number": <問題番号>,
                "Question": <質問文>,
                "Answer": <回答文>
              }, ...
            ]
    """
    with open(output_txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 正規表現で問題ごとのセクションを取得
    pattern_question = r'==========(\d+)\.Question===========\n(.*?)\n---------Answer---------\n(.*?)(?=\n==========|$)'
    matches = re.findall(pattern_question, text, re.DOTALL)

    results_list = []
    for match in matches:
        question_number = match[0]
        question_text = match[1].strip()
        answer_text = match[2].strip()
        results_list.append({
            "Question Number": question_number,
            "Question": question_text,
            "Answer": answer_text
        })
    return results_list


def get_answers_from_exploited(step: int, results: list) -> str:
    """
    read_output_fileで取得したリストから、該当ステップ(問題番号)の回答を取り出す関数。
    
    Args:
        step (int): 何問目かを示すインデックス(1始まり)
        results (list): read_output_fileで生成したリスト

    Returns:
        str: 指定ステップの回答 (LLM_Outputs)
    """
    # step-1 で添字を合わせる
    LLM_Outputs = results[step - 1]["Answer"]
    return LLM_Outputs


def make_input(
    LLM_output: str,
    question: str,
    correct_text: str,
    eval_aspect: str,
    template_path: str = './inputs/prompt_template.txt'
) -> str:
    """
    ファイルからテンプレートを読み込み、埋め込みを行って
    採点モデルに与えるプロンプト文字列を生成する関数。

    Args:
        LLM_output (str): 被評価モデルの回答
        question (str): 質問文
        correct_text (str): 正解（模範解答）
        eval_aspect (str): 採点ポイントなどの追加情報
        template_path (str): テンプレートファイルのパス

    Returns:
        str: テンプレートに情報を埋め込んだ文字列
    """
    with open(template_path, 'r', encoding='utf-8') as file:
        template = file.read()

    exam_text = template.format(
        LLM_output=LLM_output,
        question=question,
        Correct_text=correct_text,
        eval_aspect=eval_aspect
    )
    return exam_text


def evaluate_llm(
    call_in: str,
    step: int,
    critc_file_path: str,
    evaluation_model_obj: Runnable,
    evaluation_name: str
) -> str:
    """
    採点モデルにプロンプトを与えて思考させ、さらにその思考結果を再度入力して、
    「数字のみの採点結果」を取得する関数。

    RunnableParallelを用いて:
      - 第1段階: call_in を human メッセージとして入力 → モデル思考を得る
      - 第2段階: "数字のみを出力して" と命令 → モデル回答を得る

    Args:
        call_in (str): 採点用のプロンプト
        step (int): 何問目か
        critc_file_path (str): ログを書き出すファイルのパス
        evaluation_model_obj (Runnable): 採点者モデルのオブジェクト
        evaluation_name (str): 採点者モデルの名前("Google","Azure"等)

    Returns:
        str: モデルが出力した数字のみの結果(採点)
    """
    prompt1 = ChatPromptTemplate.from_messages([
        ("human", "{user_input}")
    ])
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "思考の結果以下の回答が得られたため、結論である採点結果を数字のみで出力してください。"),
        ("human", "{llm_output}")
    ])

    output_parser = StrOutputParser()
    chain1 = prompt1 | evaluation_model_obj | output_parser
    chain2 = prompt2 | evaluation_model_obj | output_parser

    chain = (
        RunnableParallel(
            {
                "user_input": RunnablePassthrough(),
                "llm_output": chain1,
            }
        )
        .assign(Answer=chain2)
    )

    max_retries = 3
    output = None
    for attempt in range(max_retries):
        try:
            output = chain.invoke({"user_input": str(call_in)})
            if evaluation_name == "Google":
                # Googleはレスポンスに時間がかかる場合があるので少し待つ
                time.sleep(20)
            break
        except Exception as e:
            print("予期しないエラーが発生しました:", e)
            if attempt < max_retries - 1:
                print("再試行します")
                time.sleep(5)
            else:
                print("再試行しましたがエラーが続いたため、処理を中断します")
                raise ValueError("処理が中断されました")

    if output is None:
        return "採点結果取得に失敗しました"

    # critc_fileにログとして書き出し
    with open(critc_file_path, mode='a', encoding="utf-8") as f:
        f.write(f"=========={step}.Prompt===========\n\n" + output["user_input"]["user_input"]
                + f"\n\n---------LLM Output---------\n\n")
        f.write(output["llm_output"]
                + f"\n\n---------LLM Answer---------\n\n" + output["Answer"] + "\n\n")

    return output["Answer"]


def score_sum(
    result_file_path: str,
    score_file_path: str,
    safe_target_model_name: str,
    safe_evaluation_model_name: str
):
    """
    各問題のスコアを集計して平均を算出し、スコアファイルに追記する関数。
    0 未満は 0、5 より大きい場合は 5 にクリップしてから平均値を出す。

    Args:
        result_file_path (str): 各問題の結果(数字)を1行ずつまとめたファイルパス
        score_file_path (str): 最終的な平均スコアを書き込むファイルパス
        safe_target_model_name (str): ファイル名向けに安全化されたTargetモデル名
        safe_evaluation_model_name (str): ファイル名向けに安全化されたEvaluationモデル名
    """
    with open(result_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines_rstrip = [line.rstrip("\n") for line in lines]
    count_lines = len(lines_rstrip)
    total_score = 0

    for line in lines_rstrip:
        val = int(line)
        if val < 0:
            val = 0
        elif val > 5:
            val = 5
        total_score += val

    avg_score = total_score / count_lines if count_lines != 0 else 0

    print(f"target_model: {safe_target_model_name}, evaluation_model: {safe_evaluation_model_name}")
    print(f"スコアは {avg_score} です")

    with open(score_file_path, mode='a', encoding="utf-8") as f:
        f.write("スコアは" + str(avg_score) + "\n")


def remove_whitespace(text: str) -> str:
    """
    文字列からスペース、改行、タブなどの空白文字を削除する。

    Args:
        text (str): 入力文字列。

    Returns:
        str: 空白文字が削除された文字列。
    """
    return re.sub(r"\s+", "", text)


def combine_files(
    score_file_path: str,
    output_file_path: str,
    result_file_path: str,
    critc_file_path: str,
    csv_file_path: str,
    markdown_output_path: str
):
    """
    最終的にMarkdown形式で結果を出力する関数。
    1. 平均スコアを取得
    2. output_fileをもとに問題数や回答情報取得
    3. critc_file(採点モデルの思考ログ)を抽出
    4. CSVファイルから模範解答や採点基準
    5. すべてをマージしてMarkdownファイルに書き出す

    Args:
        score_file_path (str): scoreファイルのパス
        output_file_path (str): 被評価モデルの出力ファイルのパス
        result_file_path (str): 数値スコアが1行ずつ書かれているファイルのパス
        critc_file_path (str): 採点モデルのログファイル
        csv_file_path (str): CSVファイル（[質問,正解,採点基準] 形式）
        markdown_output_path (str): 最終的なMarkdown出力先
    """
    # 1. 平均スコアファイルを読み込む
    with open(score_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        last_line = lines[-1] if lines else ""  # 空ファイル対応

    # 2. 被評価モデルの出力ファイルを読み込む
    with open(output_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 3. 採点モデルの出力(思考ログ)を読み込む
    with open(critc_file_path, 'r', encoding='utf-8') as file:
        critc_content = file.read()

    # outputファイルからの問題セクション抽出
    pattern_question = r'==========(\d+)\.Question===========\n(.*?)\n---------Answer---------\n(.*?)(?=\n==========|$)'
    matches = re.findall(pattern_question, text, re.DOTALL)
    print(f"問題数: {len(matches)}")

    # 採点モデルのログファイルから抽出
    prompts = re.findall(r'(==========\d+\.Prompt===========.*?---------LLM Output---------)', critc_content, re.DOTALL)
    numbers = re.findall(r"==========(\d+)\.Prompt==========", critc_content)
    llm_outputs = re.findall(r'---------LLM Output---------\s*(.*?)\s*---------LLM Answer---------', critc_content, re.DOTALL)
    answers = re.findall(r'(---------LLM Answer---------.*?==========\d+\.Prompt===========)', critc_content + '==========101.Prompt===========', re.DOTALL)
    print(f"Prompt数: {len(prompts)}")
    print(f"Number数: {len(numbers)}")
    print(f"LLM Output数: {len(llm_outputs)}")
    print(f"Answer数: {len(answers)}")

    # 結果をまとめる
    results = []
    for step, match in enumerate(matches):
        question_number = match[0]
        question_text = match[1].strip()
        answer_text = match[2].strip()

        pronpt = prompts[step]
        number = numbers[step]
        llm_output = llm_outputs[step]
        answer = answers[step]

        results.append({
            "Question Number": question_number,
            "Question": question_text,
            "Answer": answer_text,
            "Prompt": pronpt,
            "Number": number,
            "LLM Output": llm_output,
            "Scoring": answer
        })

    # 各問題のスコアを読み込み
    with open(result_file_path, 'r', encoding='utf-8') as f:
        scores = f.read().splitlines()

    # CSVファイルを読み込み ( [question, correct, aspect] )
    model_answers = []
    grading_criteria = []
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                model_answers.append(row[1])
                grading_criteria.append(row[2])

    # Markdown化
    markdown_content = f"# 平均スコア\n\n{last_line}\n"
    markdown_content += "# 結果と回答\n\n"

    # resultsごとに整形
    for idx, result in enumerate(results):
        # 問題番号の整合チェック
        if idx + 1 != int(remove_whitespace(result["Question Number"])):
            print(f"output-fileの問題番号が一致しません: {idx+1} != {result['Question Number']}")
            raise ValueError(f"output-fileの問題番号が一致しません: step:{idx+1} != file:{result['Question Number']}")

        if idx + 1 != int(remove_whitespace(result["Number"])):
            print(f"cretical-fileの問題番号が一致しません: {idx+1} != {result['Number']}")
            raise ValueError(f"cretical-fileの問題番号が一致しません: step:{idx+1} != file:{result['Number']}")

        # CSVファイルの模範解答と採点基準
        if idx < len(model_answers):
            model_answer = model_answers[idx + 1]
        else:
            model_answer = "模範解答なし"

        if idx < len(grading_criteria):
            criteria = grading_criteria[idx + 1]
        else:
            criteria = "採点基準なし"

        markdown_content += f"## 第{idx + 1}問 (点数: {scores[idx]})\n\n"
        markdown_content += f"{result['Question'].strip()}\n\n"
        markdown_content += f"### LLM出力結果:\n{result['Answer']}\n\n"
        markdown_content += f"### 模範解答:\n{model_answer}\n\n"
        markdown_content += f"### 採点基準:\n{criteria}\n\n"
        markdown_content += f"### 採点理由:\n```\n{result['LLM Output']}\n```\n\n"
        markdown_content += "---\n\n"

    with open(markdown_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"マークダウン形式のファイルが {markdown_output_path} に保存されました。")


def main():
    """
    メインの処理をまとめた関数。
    1. HuggingFace ログイン
    2. ファイルパス系を整理
    3. 評価モデルの初期化
    4. output_txt の読み込み → CSVとの付き合わせ
    5. 採点結果をファイルに追記
    6. スコア集計＆Markdown出力
    """

    #-------------------------
    # 1. HuggingFaceログイン
    #-------------------------
    if HuggingFace_access:
        from huggingface_hub import login
        login(token=os.getenv("HF_TOKEN", ""))

    #-------------------------
    # 2. ファイルパス系
    #-------------------------
    file_paths = get_file_paths(output_txt, Evaluation, Evaluation_model)
    result_file_path = file_paths["result_file"]
    critc_file_path = file_paths["critc_file"]
    score_file_path = file_paths["score_file"]
    markdown_output_path = file_paths["markdown_output"]
    safe_target_model_name = file_paths["safe_target_model"]
    safe_evaluation_model_name = file_paths["safe_evaluation_model"]

    #-------------------------
    # 3. 評価モデルの初期化
    #-------------------------
    model_obj = initialize_evaluation_model(
        Evaluation,
        Evaluation_model,
        Evaluation_temperature,
        Evaluation_top_p
    )

    #-------------------------
    # 4. output_txt の読み込み
    #-------------------------
    results_list = read_output_file(output_txt)

    #-------------------------
    # CSVを読み込みながら採点処理
    #-------------------------
    count = 0
    with open(csv_file, 'r', encoding='utf-8') as csvfile_obj:
        reader = csv.reader(csvfile_obj)
        for step, row in enumerate(reader):
            if count > resume_question_index - 1:
                print(f"問題: {row[0]}, 回答: {row[1]}, 採点ポイント: {row[2]}")

                # 被評価モデルの回答(既にファイルに書かれている)を取得
                out = get_answers_from_exploited(step, results_list)

                # テンプレートを埋め込み
                exam_text = make_input(out, row[0], row[1], row[2])

                # 採点モデルで評価を実施
                res = evaluate_llm(
                    exam_text,
                    step,
                    critc_file_path,
                    model_obj,
                    Evaluation
                )

                # 空白除去
                res_cleaned = remove_whitespace(res)

                # 数値結果をファイルに書き込む
                with open(result_file_path, mode='a', encoding="utf-8") as f:
                    f.write(str(res_cleaned) + "\n")

            else:
                count += 1
                print(count)

    #-------------------------
    # 6. スコア集計 & マークダウン出力
    #-------------------------
    score_sum(
        result_file_path,
        score_file_path,
        safe_target_model_name,
        safe_evaluation_model_name
    )
    combine_files(
        score_file_path,
        output_txt,
        result_file_path,
        critc_file_path,
        csv_file,
        markdown_output_path
    )


# メイン処理の呼び出し
if __name__ == "__main__":
    main()
