with open("result.txt","r", encoding="utf-8") as f:
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

print("スコアは"+str(score)+"です")