#!pip install langgraph langchain_openai langchain_core

import csv
import torch
import os
import re

import time
from openai import AzureOpenAI
import openai

from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.base import Runnable

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_google_genai import ChatGoogleGenerativeAI



#=================Parameter===========================================================================

#評価者モデルの選択
#Evaluation = "Azure"
#Evaluation_model = "gpt-4o"
Evaluation = "Google"
Evaluation_model = "gemini-1.5-flash"
#Evaluation = "HuggingFace"
#Evaluation_model = "meta-llama/Llama-3.3-70B-Instruct"

#評価対象のモデルの選択
#Target = "Azure"
#Target_model = "gpt-4o-mini"
#Target = "Google"
#Target_model = "gemini-1.5-flash"
Target = "HuggingFace"
Target_model = "meta-llama/Llama-3.2-1B-Instruct"

#何問目から再開するか 1問目から始める場合は1
resume_question_index = 1

#Huggingfaceにて、アクセス権限がないと取得できないモデルを利用するかどうかのフラグ
HuggingFace_access = True


#評価側温度の設定 評価に使うので、基本的に0でいい
Evaluation_temperature = 0
#評価対象側温度の設定 評価に使うので、基本的に0でいい
Target_temperature = 0

#AWSなどでキャッシュディレクトリを利用する場合はここに指定する。使わない場合はNone
efs_cache_dir = None


#=======================================================================================================


if HuggingFace_access:
    from huggingface_hub import login
    login(token=os.getenv("HF_TOKEN", ""))


#評価用のモデルを定義する
if Evaluation == "Azure":
    #環境変数を登録するもしくは、下手打ちでも良い
    #os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
    #os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxxx.openai.azure.com"
    #os.environ["AZURE_OPENAI_API_KEY"] = "AtNixxxxxxxxxxxxxxxxxxxxx"
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")

    model = AzureChatOpenAI(
        azure_deployment=Evaluation_model,
        temperature=Evaluation_temperature,
    )

elif Evaluation == "Google":
    #環境変数を登録するもしくは、下手打ちでも良い
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

    model = ChatGoogleGenerativeAI(
        model=Evaluation_model,
        temperature=Evaluation_temperature
    )

elif Evaluation == "HuggingFace":

    do_sample = False
    if Evaluation_temperature > 0:
        do_sample = True

    # モデルのロード
    llama = AutoModelForCausalLM.from_pretrained(
        Evaluation_model,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=efs_cache_dir,  # モデルをキャッシュする場所を指定
        force_download=False,
        trust_remote_code=True
        
    )

    dtypes_llama = {param.dtype for param in llama.parameters()}
    print(f"モデルで使用されているデータ型: {dtypes_llama}")

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(
        Evaluation_model,
        cache_dir=efs_cache_dir , # トークナイザーも同様にキャッシュ
        force_download=False,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.chat_template == None:
        with open("./inputs/llama_chat_template", 'r', encoding='utf-8') as file:
            template = file.read()
        tokenizer.chat_template = template


    pipe = pipeline(
        "text-generation", model=llama, tokenizer=tokenizer, temperature = Evaluation_temperature, do_sample=do_sample, max_new_tokens=1024
        
    )

    pipe = HuggingFacePipeline(pipeline=pipe)
    model = ChatHuggingFace(llm=pipe)

else:
    model = None
    print("モデルが選択されていません。")
    exit()

#評価対象のモデルを定義する
if Target == "Azure":
    #環境変数を登録するもしくは、下手打ちでも良い
    #os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
    #os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxxx.openai.azure.com"
    #os.environ["AZURE_OPENAI_API_KEY"] = "AtNixxxxxxxxxxxxxxxxxxxxx"
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
    
    llm_api = AzureChatOpenAI(
        azure_deployment=Target_model,
        temperature=Target_temperature,
    )
elif Target == "Google":
    #環境変数を登録するもしくは、下手打ちでも良い
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
    llm_api = ChatGoogleGenerativeAI(
        model=Target_model,
        temperature=Target_temperature
    )
elif Target == "HuggingFace":

    do_sample = False
    if Target_temperature > 0:
        do_sample = True

    # モデルのロード
    llama_target = AutoModelForCausalLM.from_pretrained(
        Target_model,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=efs_cache_dir,  # モデルをキャッシュする場所を指定
        force_download=False,
        trust_remote_code=True
        
    )

    dtypes_llama = {param.dtype for param in llama_target.parameters()}
    print(f"モデルで使用されているデータ型: {dtypes_llama}")

    # トークナイザーのロード
    tokenizer_target = AutoTokenizer.from_pretrained(
        Target_model,
        cache_dir=efs_cache_dir , # トークナイザーも同様にキャッシュ
        force_download=False,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer_target.chat_template == None:
        with open("./inputs/llama_chat_template", 'r', encoding='utf-8') as file:
            template = file.read()
        tokenizer_target.chat_template = template


    pipe_target = pipeline(
        "text-generation", model=llama_target, tokenizer=tokenizer_target, temperature = Target_temperature, do_sample=do_sample, max_new_tokens=1024
    )

    pipe_target = HuggingFacePipeline(pipeline=pipe_target)
    llm_api = ChatHuggingFace(llm=pipe_target)
else:
    model = None
    print("モデルが選択されていません。")
    exit()

def sanitize_filename(filename: str) -> str:
    """
    ファイル名に使用できない文字を安全に置き換える。

    Args:
        filename (str): 元のファイル名。

    Returns:
        str: 安全なファイル名。
    """
    # 不適切な文字をハイフンまたはアンダースコアに置き換える
    sanitized = re.sub(r'[\/:*?"<>|]', '-', filename)
    return sanitized


safe_target_model = sanitize_filename(Target_model)
safe_evaluation_model = sanitize_filename(Evaluation_model)
os.makedirs(f"./outputs/{safe_target_model}", exist_ok=True)


def Evaluate_LLM(call_in:str, step:int):
    prompt1 = ChatPromptTemplate.from_messages(
        [
            ("human", "{user_input}")
        ]
    )
    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system","思考の結果以下の回答が得られたため、結論である採点結果を数字のみで出力してください。"),
            ("human", "{llm_output}")
        ]
    )


    output_parser = StrOutputParser()

    #ここでChainを組む
    chain1 = prompt1 | model | output_parser
    chain2 = prompt2 | model | output_parser

    chain = (
        RunnableParallel(
        {
            "user_input": RunnablePassthrough(),
            "llm_output": chain1,
        }
    )
    .assign(Answer=chain2)
    )

    output = chain.invoke({"user_input":str(call_in)})
    #print(output)
    with open(f"./outputs/{safe_target_model}/cretical-{Target}-{safe_target_model}_by_{safe_evaluation_model}.txt", mode='a', encoding="utf-8") as f:
        f.write(f"=========={step}.Prompt===========\n\n" + output["user_input"]["user_input"] +f"\n\n---------LLM Output---------\n\n")
        f.write(output["llm_output"] +f"\n\n---------LLM Answer---------\n\n" + output["Answer"] + "\n\n")
    return output["Answer"]



def get_LLM_chain(
    query: Runnable, 
    llm_api: Runnable, 
    output_parser: Runnable
) -> Runnable:
    """
    LLMチェーンを生成する関数。

    Args:
        query (Runnable): 入力プロンプトを生成する `Runnable` オブジェクト。
        llm_api (Runnable): 言語モデルへのAPI呼び出しを実行する `Runnable` オブジェクト。
        output_parser (Runnable): モデル出力を処理する `Runnable` オブジェクト。

    Returns:
        Runnable: 組み合わせたLLMチェーン。
    """
    if Target == "HuggingFace":
        return query | llm_api.bind(skip_prompt=True) | output_parser
    else:
        return query | llm_api | output_parser
       


def Answers_LLM(user_inputs_text:str):
    messages_api = [
        ("system","あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        ("human", "{user_input}")
    ]
    
    query = ChatPromptTemplate.from_messages(messages_api)
    output_parser = StrOutputParser()
    print(query.invoke({"user_input":user_inputs_text}))
    #ここでChainを組む
    chain = get_LLM_chain(query, llm_api, output_parser)

    max_retries = 3
    for attempt in range(max_retries):
        try:

            response_api = chain.invoke({"user_input":user_inputs_text})
            
            return response_api
        
        except openai.BadRequestError as e:
            print("エラーを感知しました")
            print(e)
            if 'content_filter' in str(e):
                print("不適切なワードを感知しました")
                return "不適切なワードを感知したため実行できませんでした"
            else:
                if attempt < max_retries - 1:
                    print("再試行します")
                    time.sleep(5)  # 5秒待機して再試行
                else:
                    print("再試行しましたがエラーが続いたため、処理を中断します")

    return "再試行しましたがエラーが続いたため、処理を中断します"


    


def make_input(LLM_output:str, question:str, Correct_text:str, eval_aspect:str, template_path:str ='./inputs/prompt_template.txt'):
    # ファイルからテンプレートを読み込む
    with open(template_path, 'r', encoding='utf-8') as file:
        template = file.read()
    
    # テンプレート内のプレースホルダに引数を埋め込む
    exam_text = template.format(
        LLM_output=LLM_output,
        question=question,
        Correct_text=Correct_text,
        eval_aspect=eval_aspect
    )
    return exam_text

def score_sum():
    with open(f"./outputs/{safe_target_model}/result-{Target}-{safe_target_model}_by_{safe_evaluation_model}.txt","r", encoding="utf-8") as f:
        d = f.readlines()

    lines_rstrip_D = [lineD.rstrip("\n") for lineD in d]
    comment_count_D = len(lines_rstrip_D)

    result = 0

    for i in lines_rstrip_D:
        i = int(i)
        if i < 0:
            i = 0
        elif i> 5:
            i = 5
        result = result + int(i)

    score = result/comment_count_D

    print(f"target_model: {safe_target_model}, evaluation_model: {safe_evaluation_model}")
    print("スコアは"+str(score)+"です")

    with open(f"./outputs/{safe_target_model}/score-{Target}-{safe_target_model}_by_{safe_evaluation_model}.txt", mode='a', encoding="utf-8") as f:
        f.write("スコアは"+str(score)+"です\n")

def combine_files(output_file:str, result_file:str, csv_file:str, markdown_output:str):
    # Load questions and answers
    with open(output_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 正規表現で問題ごとのセクションを取得
    pattern_question = r'==========(\d+)\.Question===========\n(.*?)\n---------Answer---------\n(.*?)(?=\n==========|$)'
    matches = re.findall(pattern_question, text, re.DOTALL)

    # 結果を格納
    results = []

    # 各問題と回答を処理
    for match in matches:
        question_number = match[0]  # 問題番号
        question_text = match[1].strip()  # 質問部分
        answer_text = match[2].strip()  # 回答部分
        results.append({
            "Question Number": question_number,
            "Question": question_text,
            "Answer": answer_text
        })
        
    # Load scores
    with open(result_file, 'r', encoding='utf-8') as f:
        scores = f.read().splitlines()
    
    # Load model answers and criteria
    model_answers = []
    grading_criteria = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:  # Ensure there are at least model answer and grading criteria
                model_answers.append(row[1])
                grading_criteria.append(row[2])

    # Create the markdown content
    markdown_content = "# 結果と回答\n\n"
    
    for idx, result in enumerate(results):
        model_answer = model_answers[idx+1] if idx < len(model_answers) else "模範解答なし"
        criteria = grading_criteria[idx+1] if idx < len(grading_criteria) else "採点基準なし"
        
        markdown_content += f"## 第{idx + 1}問 (点数: {scores[idx]})\n\n"
        markdown_content += f"{result['Question'].strip()}\n\n"
        markdown_content += f"### LLM出力結果:\n{result['Answer']}\n\n"
        markdown_content += f"### 模範解答:\n{model_answer}\n\n"
        markdown_content += f"### 採点基準:\n{criteria}\n\n"
        markdown_content += "---\n\n"
    
    with open(markdown_output, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"マークダウン形式のファイルが {markdown_output} に保存されました。")


def remove_whitespace(text: str) -> str:
    """
    文字列からスペース、改行、タブなどの空白文字を削除する。

    Args:
        text (str): 入力文字列。

    Returns:
        str: 空白文字が削除された文字列。
    """
    return re.sub(r"\s+", "", text)


output_file = f"./outputs/{safe_target_model}/output-{Target}-{safe_target_model}_by_{safe_evaluation_model}.txt"
result_file = f"./outputs/{safe_target_model}/result-{Target}-{safe_target_model}_by_{safe_evaluation_model}.txt"
csv_file = './inputs/test.csv'
markdown_output = f"./outputs/{safe_target_model}/Elyza-{Target}-{safe_target_model}_by_{safe_evaluation_model}.md"




count = 0

with open(csv_file, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for step, row in enumerate(reader):
        if count > resume_question_index - 1:
            print(f'問題: {row[0]}, 回答: {row[1]}, 採点ポイント: {row[2]}')
            out = Answers_LLM(str(row[0]))
            
            
            with open(output_file, mode='a', encoding="utf-8") as f:
                f.write(f"=========={step}.Question===========\n\n" + str(row[0]) +f"\n\n---------Answer---------\n\n")
                f.write(str(out) + "\n\n")

            exam_text = make_input(out,row[0],row[1],row[2])
            res = Evaluate_LLM(exam_text, step)
            res = remove_whitespace(res)
            with open(result_file, mode='a', encoding="utf-8") as f:
                f.write(str(res) +"\n")
        else:
            count = count + 1
            print(count)
    
score_sum()
combine_files(output_file, result_file, csv_file, markdown_output)


