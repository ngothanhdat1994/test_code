import os

embed_list = []

for i in range (512):
    i +=1
    name = ",embed" + str(i)
    embed_list.append(name)

with open(os.path.join("trill_embeddings"+".csv"), "a") as text_file:
        text_file.write("file_ID"+ ''.join(embed_list) + "\n")
    