__author__ = 'pier-luc plante'


import sys

currentCluster = ""
clusterDict = {}
samplesName=[]

for line in open(sys.argv[1]):
    if line == "":
        break
    if line[0] == '>':
        entryName = line.split()[0].split('>')[1]+'_'+line.split()[1]
        clusterDict[entryName] = []
        currentCluster = entryName
    else:
        sample_id = (line.split()[2].split('|')[0])[1:]
        ##clusterDict[currentCluster] += sample_id;
        clusterDict[entryName].append(sample_id)
        if sample_id not in samplesName:
            samplesName.append(sample_id)


#to throw the info in a .tsv format to stdout
line="ClusterID"
samplesName.sort()
for name in samplesName:
    line = line+'\t'+name
print(line)

for entry in clusterDict:
    line_start = str(entry)
    line_content = ""
    line_total = 0
    for name in samplesName:
        if name in clusterDict[entry]:
            line_content = line_content +'\t' + str(clusterDict[entry].count(name))
            line_total += 1
        else:
            line_content += '\t0'
    line = line_start + line_content
    print(line)