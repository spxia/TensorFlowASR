import os
dir="/tsdata/ASR/aishell-1/wav/test"
out="aaa.txt"
type="test"
wav_files = []
"""
with open(out, "w", encoding="utf-8") as output:
    for root, dirs, files in os.walk(dir):
        for file in files:
            (filename,extension) = os.path.splitext(file)
            #wav_files[filename] = os.path.join(root,file)
            if "8k" not in filename:
                wav_files.append(filename + "\t" + os.path.join(root, file))
    wav_files.sort()
    for line in wav_files:
        output.write(line + "\n")
"""
lista=[]
with open("/tsdata/ASR/aishell-1/transcript/transcripts.txt","r",encoding="utf-8") as txt:
    lines = txt.read().splitlines()
    for line in lines:
        line = line.split(" ",maxsplit=1)
        for i in line[1]:
            lista.append(i)
a = list(set(lista))
a.sort()

for i in a:
    print (i)

#for (key,value) in wav_files.items():
#    print(key+':'+value)