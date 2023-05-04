#extracting sentence pairs and labels from the PAWS dataset

import csv
with open("train.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    #c=0
    id=[]
    sentence1=[]
    sentence2=[]
    label_assigned=[]
    for line in tsv_file:
        id.append(line[0])
        sentence1.append(line[1])
        sentence2.append(line[2])
        label_assigned.append(line[3])
    print(len(sentence1))
    sentence3=[]
    sentence4=[]
    for i in range(0,len(label_assigned)):
        if label_assigned[i]=='1':
            sentence3.append(sentence1[i])
            sentence4.append(sentence2[i])
    # with open('sentences_in.txt', 'w') as filehandle:
    #     for listitem in sentence1:
    #         filehandle.write(f'{listitem}\n')
    # with open('sentences_out.txt', 'w') as filehandle:
    #     for listitem in sentence2:
    #         filehandle.write(f'{listitem}\n')
    print(len(sentence3))
    with open('paraphrases_in.txt', 'w') as filehandle:
        for listitem in sentence3:
            filehandle.write(f'{listitem}\n')
    with open('paraphrases_out.txt', 'w') as filehandle:
        for listitem in sentence4:
            filehandle.write(f'{listitem}\n')
    print("Done")