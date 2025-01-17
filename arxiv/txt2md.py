import csv
import re

def combine_files(output_file, result_file, csv_file, markdown_output):
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

# ファイルパスを指定して実行
#output_file = './outputs/output-4omini.txt'
#result_file = './outputs/result-4omini.txt'
#output_file = './outputs/output-Llama-3.3-70B-Instruct-1024.txt'
#result_file = './outputs/result-Llama-3.3-70B-Instruct-1024.txt'
#output_file = './outputs/output-Llama-3.1-8B-Instruct.txt'
#result_file = './outputs/result-Llama-3.1-8B-Instruct.txt'
#output_file = './outputs/output-Llama-3.1-70B-Instruct.txt'
#result_file = './outputs/result-Llama-3.1-70B-Instruct.txt'
output_file = './outputs/output-Linkbricks-Horizon-AI-Japanese-Pro-V4-70B.txt'
result_file = './outputs/result-Linkbricks-Horizon-AI-Japanese-Pro-V4-70B.txt'
csv_file = './inputs/test.csv'
markdown_output = './outputs/combined_output.md'
combine_files(output_file, result_file, csv_file, markdown_output)
