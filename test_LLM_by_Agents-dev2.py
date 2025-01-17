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
# (1) ここで宣言しているパラメータはグローバル変数として扱います。
#     これ以外の変数はグローバルスコープに置かないようにします。

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

# 評価対象のモデルの選択
# "OpenAI_Base"では、gpt-4o系統もしくは、deepseekのみ実装済み
Target = "OpenAI_Base"
Target_model = "deepseek-chat"
#Target = "OpenAI_Base"
#Target_model = "gpt-4o-mini"
#Target = "Azure"
#Target_model = "gpt-4o"
#Target = "Google"
#Target_model = "gemini-1.5-flash" #"gemini-2.0-flash-exp", "gemini-1.5-flash"
#Target = "HuggingFace"
#Target_model = "meta-llama/Llama-3.2-1B-Instruct"

# 何問目から再開するか（1問目から始める場合は1）
resume_question_index = 1

# HuggingFaceにて、アクセス権限がないと取得できないモデルを利用するかどうかのフラグ
HuggingFace_access = True

# 採点側温度の設定（評価に使うので、基本的に0.001でいい）
Evaluation_temperature = 0.001
# 採点側top_pの設定（評価に使うので、基本的に0.001でいい）
Evaluation_top_p = 0.001

# 評価対象側温度の設定（評価に使うので、基本的に0.001でいい）
Target_temperature = 0.001
# 評価対象側top_pの設定（評価に使うので、基本的に0.001でいい）
Target_top_p = 0.001

# AWSなどでキャッシュディレクトリを利用する場合はここに指定する。使わない場合はNone
efs_cache_dir = None

# CSVファイルのパス
csv_file = './inputs/test.csv'
#=======================================================================================================


def sanitize_filename(filename: str) -> str:
    """
    ファイル名に使用できない文字を安全に置き換える関数。

    Args:
        filename (str): 元のファイル名

    Returns:
        str: 不正文字がハイフンに置き換えられた安全なファイル名
    """
    sanitized = re.sub(r'[\/:*?"<>|]', '-', filename)
    return sanitized


def get_file_paths(
    target: str,
    target_model: str,
    evaluation: str,
    evaluation_model: str
) -> dict:
    """
    出力ファイルのパスを生成する関数。
    ファイル名に使用できない文字を含む場合はsanitize_filenameで安全化した上で
    ディレクトリを作成し、パスを生成します。

    Args:
        target (str): 評価対象モデル名
        target_model (str): 評価対象モデルの詳細名
        evaluation (str): 評価モデル名
        evaluation_model (str): 評価モデルの詳細名

    Returns:
        dict: 各ファイルのパスを格納した辞書
    """
    safe_target_model = sanitize_filename(target_model)
    safe_evaluation_model = sanitize_filename(evaluation_model)
    output_dir = f"./outputs/{target}-{safe_target_model}"
    os.makedirs(output_dir, exist_ok=True)

    return {
        "output_file": f"{output_dir}/output-{target}-{safe_target_model}.txt",
        "result_file": f"{output_dir}/result-{target}-{safe_target_model}_by_{safe_evaluation_model}.txt",
        "critc_file": f"{output_dir}/cretical-{target}-{safe_target_model}_by_{safe_evaluation_model}.txt",
        "score_file": f"{output_dir}/score-{target}-{safe_target_model}_by_{safe_evaluation_model}.txt",
        "markdown_output": f"{output_dir}/Elyza-{target}-{safe_target_model}_by_{safe_evaluation_model}.md",
        "safe_target_model": safe_target_model,
        "safe_evaluation_model": safe_evaluation_model,
    }


def initialize_evaluation_model(
    evaluation_name: str,
    evaluation_model_name: str,
    evaluation_temperature: float,
    evaluation_top_p: float
):
    """
    採点に使用するモデルを初期化して返す関数。

    Args:
        evaluation_name (str): 評価モデル名（"Azure", "Google", "HuggingFace", "OpenAI_Base"等）
        evaluation_model_name (str): 評価モデルの詳細名
        evaluation_temperature (float): 温度パラメータ
        evaluation_top_p (float): top_pパラメータ

    Returns:
        model: 評価モデルとして使用するLLMオブジェクト
    """
    model = None

    if evaluation_name == "Azure":
        # 環境変数を登録するもしくは、直書きでも良い
        # os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
        # os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxxx.openai.azure.com"
        # os.environ["AZURE_OPENAI_API_KEY"] = "AtNixxxxxxxxxxxxxxxxxxxxx"
        os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "")
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")

        model = AzureChatOpenAI(
            azure_deployment=evaluation_model_name,
            temperature=evaluation_temperature,
        )

    elif evaluation_name == "Google":
        # 環境変数を登録するもしくは、直書きでも良い
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

        model = ChatGoogleGenerativeAI(
            model=evaluation_model_name,
            temperature=evaluation_temperature,
            top_p=evaluation_top_p
        )

    elif evaluation_name == "HuggingFace":
        do_sample = (evaluation_temperature > 0.001)

        huggingface_model = AutoModelForCausalLM.from_pretrained(
            evaluation_model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=efs_cache_dir, # モデルをキャッシュする場所を指定
            force_download=False,
            trust_remote_code=True
        )

        dtypes_llama = {param.dtype for param in huggingface_model.parameters()}
        print(f"モデルで使用されているデータ型: {dtypes_llama}")

        tokenizer = AutoTokenizer.from_pretrained(
            evaluation_model_name,
            cache_dir=efs_cache_dir, # トークナイザーも同様にキャッシュ
            force_download=False,
            trust_remote_code=True,
            use_fast=True
        )

        # LLaMa系の場合にテンプレートを読み込む
        if (tokenizer.chat_template is None) and ("llama" in evaluation_model_name.lower()):
            with open("./inputs/llama_chat_template.txt", 'r', encoding='utf-8') as file:
                template = file.read()
            tokenizer.chat_template = template

        pipe = pipeline(
            "text-generation",
            model=huggingface_model,
            tokenizer=tokenizer,
            temperature=evaluation_temperature,
            do_sample=do_sample,
            max_new_tokens=1024
        )
        pipe = HuggingFacePipeline(pipeline=pipe)
        model = ChatHuggingFace(llm=pipe, tokenizer=pipe.pipeline.tokenizer)

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

        model = ChatOpenAI(
            model=evaluation_model_name,
            openai_api_key=API_KEY,
            openai_api_base=ENDPOINT,
            max_tokens=4096,
            temperature=evaluation_temperature,
            top_p=evaluation_top_p,
            stream=False
        )
    else:
        print("モデルが選択されていません。")
        exit()

    return model


def initialize_target_model(
    target_name: str,
    target_model_name: str,
    target_temperature: float,
    target_top_p: float
):
    """
    評価対象のモデルを初期化して返す関数。

    Args:
        target_name (str): 評価対象モデル名（"Azure", "Google", "HuggingFace", "OpenAI_Base"等）
        target_model_name (str): 評価対象モデルの詳細名
        target_temperature (float): 温度パラメータ
        target_top_p (float): top_pパラメータ

    Returns:
        llm_api: 評価対象モデルとして使用するLLMオブジェクト
    """
    llm_api = None

    if target_name == "Azure":
        # 環境変数を登録するもしくは、直書きでも良い
        # os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
        # os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxxx.openai.azure.com"
        # os.environ["AZURE_OPENAI_API_KEY"] = "AtNixxxxxxxxxxxxxxxxxxxxx"
        os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "")
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")

        llm_api = AzureChatOpenAI(
            azure_deployment=target_model_name,
            temperature=target_temperature,
            top_p=target_top_p
        )

    elif target_name == "Google":
        # 環境変数を登録するもしくは、直書きでも良い
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

        llm_api = ChatGoogleGenerativeAI(
            model=target_model_name,
            temperature=target_temperature,
            top_p=target_top_p
        )

    elif target_name == "HuggingFace":
        do_sample = (target_temperature > 0.001)

        llama_target = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=efs_cache_dir,
            force_download=False,
            trust_remote_code=True
        )

        dtypes_llama_target = {param.dtype for param in llama_target.parameters()}
        print(f"モデルで使用されているデータ型: {dtypes_llama_target}")

        tokenizer_target = AutoTokenizer.from_pretrained(
            target_model_name,
            cache_dir=efs_cache_dir,
            force_download=False,
            trust_remote_code=True,
            use_fast=True
        )

        if (tokenizer_target.chat_template is None) and ("llama" in target_model_name.lower()):
            with open("./inputs/llama_chat_template.txt", 'r', encoding='utf-8') as file:
                template = file.read()
            tokenizer_target.chat_template = template

        pipe_target = pipeline(
            "text-generation",
            model=llama_target,
            tokenizer=tokenizer_target,
            temperature=target_temperature,
            do_sample=do_sample,
            max_new_tokens=1024
        )
        pipe_target = HuggingFacePipeline(pipeline=pipe_target)
        llm_api = ChatHuggingFace(llm=pipe_target, tokenizer=pipe_target.pipeline.tokenizer)

    elif target_name == "OpenAI_Base":
        API_KEY = None
        ENDPOINT = None
        if "gpt" in target_model_name:
            API_KEY = os.getenv("OPENAI_API_KEY", "")
            ENDPOINT = "https://api.openai.com/v1"
        elif target_model_name == "deepseek-chat":
            API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
            ENDPOINT = "https://api.deepseek.com"
        else:
            raise ValueError("モデルが不正です。")

        llm_api = ChatOpenAI(
            model=target_model_name,
            openai_api_key=API_KEY,
            openai_api_base=ENDPOINT,
            max_tokens=4096,
            temperature=target_temperature,
            top_p=target_top_p,
            stream=False
        )

    else:
        print("モデルが選択されていません。")
        exit()

    return llm_api


def get_LLM_chain(
    query: Runnable,
    llm_api: Runnable,
    output_parser: Runnable,
    target_name: str
) -> Runnable:
    """
    指定した query, llm_api, output_parser を組み合わせて LLM チェーンを返す関数。
    HuggingFace の場合はプロンプトをスキップする仕様に合わせてバインドを行います。

    Args:
        query (Runnable): 入力プロンプトを生成するRunnableオブジェクト
        llm_api (Runnable): 言語モデルへのAPI呼び出しを実行するRunnableオブジェクト
        output_parser (Runnable): モデル出力を処理するRunnableオブジェクト
        target_name (str): ターゲットモデル名("HuggingFace"など)

    Returns:
        Runnable: 組み合わせたLLMチェーン
    """
    if target_name == "HuggingFace":
        return query | llm_api.bind(skip_prompt=True) | output_parser
    else:
        return query | llm_api | output_parser


def Answers_LLM(
    user_inputs_text: str,
    llm_api: Runnable,
    target_name: str
) -> str:
    """
    評価対象のモデルに入力を与え、回答を得るための関数。
    トークンエラー等が出る場合は最大3回まで再試行し、不適切ワードの検知などで
    実行不可の場合はメッセージを返します。

    Args:
        user_inputs_text (str): ユーザーが入力する質問文やメッセージ
        llm_api (Runnable): 評価対象モデルのLLMオブジェクト
        target_name (str): 評価対象モデル名("HuggingFace","Google","OpenAI_Base","Azure"等)

    Returns:
        str: モデルの回答テキスト
    """
    messages_api = [
        ("system", "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        ("human", "{user_input}")
    ]
    query = ChatPromptTemplate.from_messages(messages_api)
    output_parser = StrOutputParser()
    chain = get_LLM_chain(query, llm_api, output_parser, target_name)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_api = chain.invoke({"user_input": user_inputs_text})
            return response_api

        except openai.BadRequestError as e:
            print("エラーを感知しました:", e)
            if 'content_filter' in str(e):
                print("不適切なワードを感知しました")
                return "不適切なワードを感知したため実行できませんでした"
            else:
                if attempt < max_retries - 1:
                    print("再試行します")
                    time.sleep(5)  # 5秒待機して再試行
                else:
                    print("再試行しましたがエラーが続いたため、処理を中断します")

        except Exception as e:
            print("予期しないエラーが発生しました:", e)
            if attempt < max_retries - 1:
                print("再試行します")
                time.sleep(5)  # 5秒待機して再試行
            else:
                print("再試行しましたがエラーが続いたため、処理を中断します")

    return "再試行しましたがエラーが続いたため、処理を中断します"


def make_input(
    llm_output: str,
    question: str,
    correct_text: str,
    eval_aspect: str,
    template_path: str = './inputs/prompt_template.txt'
) -> str:
    """
    採点用のプロンプトを生成するための関数。
    テンプレートファイルを読み込み、回答や採点基準などを埋め込んだ文字列を作成します。

    Args:
        llm_output (str): 評価対象のモデルの回答
        question (str): 出題された質問文
        correct_text (str): 正解（模範解答）
        eval_aspect (str): 採点基準などの補足
        template_path (str): テンプレートファイルのパス

    Returns:
        str: 生成された採点用プロンプト
    """
    with open(template_path, 'r', encoding='utf-8') as file:
        template = file.read()

    exam_text = template.format(
        LLM_output=llm_output,
        question=question,
        Correct_text=correct_text,
        eval_aspect=eval_aspect
    )
    return exam_text


def Evaluate_LLM(
    call_in: str,
    step: int,
    critc_file_path: str,
    evaluation_model_obj: Runnable,
    evaluation_name: str
) -> str:
    """
    採点者モデルを用いて入力を与え、さらにその出力を再度入力として数字のみの採点結果を取得する関数。
    RunnableParallelを用い、出力テキストとその数字のみの結論を同時取得します。

    Args:
        call_in (str): 採点用プロンプト
        step (int): 問題番号
        critc_file_path (str): 採点モデルが出した思考過程などのログを保存するファイルパス
        evaluation_model_obj (Runnable): 採点者モデルのLLMオブジェクト
        evaluation_name (str): 採点者モデル名（"HuggingFace","Google","OpenAI_Base","Azure"等）

    Returns:
        str: 採点結果（数字）
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
                # geminiのフリープランでは、RPMの制限が小さいため、待機する
                # プロプランを利用している場合などは、不要な待機時間
                time.sleep(5)
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
    各問題の採点結果を読み込み、平均スコアを計算する関数。
    5点満点（0〜5で評価）として、範囲外の得点はクリップします。

    Args:
        result_file_path (str): 各問題の採点結果（数字のみ）が1行ずつ記載されたファイルのパス
        score_file_path (str): 平均スコアを追記するファイルのパス
        safe_target_model_name (str): ファイル名向けに安全化されたTargetモデル名
        safe_evaluation_model_name (str): ファイル名向けに安全化されたEvaluationモデル名
    """
    with open(result_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    stripped_lines = [line.rstrip("\n") for line in lines]
    comment_count = len(stripped_lines)
    total_score = 0

    for line in stripped_lines:
        try:
            val = int(line)
        except ValueError:
            val = 0
        if val < 0:
            val = 0
        elif val > 5:
            val = 5
        total_score += val

    average_score = total_score / comment_count if comment_count != 0 else 0

    print(f"target_model: {safe_target_model_name}, evaluation_model: {safe_evaluation_model_name}")
    print("スコアは" + str(average_score) + "です")

    with open(score_file_path, mode='a', encoding="utf-8") as f:
        f.write("スコアは" + str(average_score) + "です\n")


def remove_whitespace(text: str) -> str:
    """
    文字列からスペース、改行、タブなどの空白文字を削除する関数。

    Args:
        text (str): 入力文字列

    Returns:
        str: 空白文字が削除された文字列
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
    最終的に平均スコア、質問、回答、採点基準、採点理由などを
    マークダウン形式にまとめて出力する関数。

    Args:
        score_file_path (str): 平均スコアが記載されたファイルのパス
        output_file_path (str): 被評価モデルの出力（各問題ごとに区切られている）のファイルのパス
        result_file_path (str): 採点モデルが出したスコア（数字のみ）が並んでいるファイルのパス
        critc_file_path (str): 採点モデルのプロンプトと思考過程を残したファイルのパス
        csv_file_path (str): CSV形式の問題・模範解答・採点基準が載っているファイルのパス
        markdown_output_path (str): 結果をマークダウン形式で出力するファイルのパス
    """
    with open(score_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        last_line = lines[-1] if lines else ""

    with open(output_file_path, 'r', encoding='utf-8') as f:
        output_text = f.read()

    with open(critc_file_path, 'r', encoding='utf-8') as file:
        critc_content = file.read()

    pattern_question = r'==========(\d+)\.Question===========\n(.*?)\n---------Answer---------\n(.*?)(?=\n==========|$)'
    matches = re.findall(pattern_question, output_text, re.DOTALL)
    print(f"問題数: {len(matches)}")

    prompts = re.findall(r'(==========\d+\.Prompt===========.*?---------LLM Output---------)', critc_content, re.DOTALL)
    numbers = re.findall(r"==========(\d+)\.Prompt==========", critc_content)
    llm_outputs = re.findall(r'---------LLM Output---------\s*(.*?)\s*---------LLM Answer---------', critc_content, re.DOTALL)
    answers = re.findall(r'(---------LLM Answer---------.*?==========\d+\.Prompt===========)', critc_content + '==========101.Prompt===========', re.DOTALL)

    print(f"Prompt数: {len(prompts)}")
    print(f"Number数: {len(numbers)}")
    print(f"LLM Output数: {len(llm_outputs)}")
    print(f"Answer数: {len(answers)}")

    results = []
    for step, match in enumerate(matches):
        question_number = match[0]
        question_text = match[1].strip()
        answer_text = match[2].strip()

        prompt_data = prompts[step]
        number_data = numbers[step]
        llm_output_data = llm_outputs[step]
        answer_data = answers[step]

        results.append({
            "Question Number": question_number,
            "Question": question_text,
            "Answer": answer_text,
            "Prompt": prompt_data,
            "Number": number_data,
            "LLM Output": llm_output_data,
            "Scoring": answer_data
        })

    with open(result_file_path, 'r', encoding='utf-8') as f:
        scores = f.read().splitlines()

    model_answers = []
    grading_criteria = []
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                model_answers.append(row[1])
                grading_criteria.append(row[2])

    markdown_content = f"# 平均スコア\n\n{last_line}\n"
    markdown_content += "# 結果と回答\n\n"

    for idx, result in enumerate(results):
        if idx + 1 != int(remove_whitespace(result["Question Number"])):
            print(f"output-fileの問題番号が一致しません: {idx+1} != {result['Question Number']}")
            raise ValueError(f"output-fileの問題番号が一致しません: step:{idx+1} != file:{result['Question Number']}")

        if idx + 1 != int(remove_whitespace(result["Number"])):
            print(f"cretical-fileの問題番号が一致しません: {idx+1} != {result['Number']}")
            raise ValueError(f"cretical-fileの問題番号が一致しません: step:{idx+1} != file:{result['Number']}")

        if idx + 1 < len(model_answers):
            model_answer = model_answers[idx + 1]
        else:
            model_answer = "模範解答なし"

        if idx + 1 < len(grading_criteria):
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
    メインの処理をまとめる関数。
    1. 各ファイルパスやモデルを初期化
    2. CSVファイルの読み込み → 評価対象モデルによる回答生成
    3. 上記回答をもとに採点モデルでスコア生成
    4. スコアの集計とMarkdown形式ファイルへのまとめ
    """

    #---------------------------------------
    # ファイルパス情報やパラメータの準備
    #---------------------------------------
    if HuggingFace_access:
        from huggingface_hub import login
        login(token=os.getenv("HF_TOKEN", ""))

    file_paths = get_file_paths(Target, Target_model, Evaluation, Evaluation_model)
    output_file = file_paths["output_file"]
    result_file = file_paths["result_file"]
    critc_file = file_paths["critc_file"]
    score_file = file_paths["score_file"]
    markdown_output = file_paths["markdown_output"]
    safe_target_model_name = file_paths["safe_target_model"]
    safe_evaluation_model_name = file_paths["safe_evaluation_model"]

    #---------------------------------------
    # モデル初期化
    #---------------------------------------
    evaluation_model_obj = initialize_evaluation_model(
        Evaluation,
        Evaluation_model,
        Evaluation_temperature,
        Evaluation_top_p
    )

    llm_api = initialize_target_model(
        Target,
        Target_model,
        Target_temperature,
        Target_top_p
    )

    #---------------------------------------
    # CSVファイルを読み込みながら回答生成・採点
    #---------------------------------------
    count = 0
    with open(csv_file, 'r', encoding='utf-8') as csvfile_obj:
        reader = csv.reader(csvfile_obj)
        for step, row in enumerate(reader):
            # row: [問題文, 正解, 採点基準]
            if count > resume_question_index - 1:
                print(f'問題: {row[0]}, 回答: {row[1]}, 採点ポイント: {row[2]}')

                # 被評価モデルへの質問
                out = Answers_LLM(str(row[0]), llm_api, Target)

                # 出力結果をファイルに書き込む
                with open(output_file, mode='a', encoding="utf-8") as f:
                    f.write(f"=========={step}.Question===========\n\n" + str(row[0])
                            + f"\n\n---------Answer---------\n\n")
                    f.write(str(out) + "\n\n")

                # 採点用の入力を組み立てる
                exam_text = make_input(out, row[0], row[1], row[2])

                # 採点モデルで採点
                res = Evaluate_LLM(
                    exam_text, step,
                    critc_file,
                    evaluation_model_obj,
                    Evaluation
                )
                # 数字以外の空白や改行記号は削除する
                res = remove_whitespace(res)

                with open(result_file, mode='a', encoding="utf-8") as f:
                    f.write(str(res) + "\n")
            else:
                count += 1
                print(count)

    #---------------------------------------
    # スコア集計＆ファイル結合
    #---------------------------------------
    score_sum(
        result_file,
        score_file,
        safe_target_model_name,
        safe_evaluation_model_name
    )

    combine_files(
        score_file,
        output_file,
        result_file,
        critc_file,
        csv_file,
        markdown_output
    )


# メイン関数の呼び出し
if __name__ == "__main__":
    main()
