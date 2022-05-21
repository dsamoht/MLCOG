# Production of orthologs clusters from shotgun metagenomics reads
__Input__: reads files from __*n*__ samples\
__Output__: __*m*__ x __*n*__ matrix (.tsv) where __*m*__ is the number of ortholog clusters. Each element corresponds to the number of orthologs in the sample (metagenome) matching the cluster's representant sequence.

## Requirements
* [MEGAHIT](https://github.com/voutcn/megahit)
* [Prodigal](https://github.com/hyattpd/Prodigal)
* [CD-HIT](https://github.com/weizhongli/cdhit)
### __1 -__ Navigate to the directory containing the reads files. 
```
├── reads_from_study
│   ├── SRR123456_1.fastq.gz
│   ├── SRR123456_2.fastq.gz
│   ├── SRR234567_1.fastq.gz
│   ├── SRR234567_2.fastq.gz
│   ├── SRR345678_1.fastq.gz
│   ├── SRR345678_2.fastq.gz
```
```
cd reads_from_study/
```
### __2 -__ You can either execute the following script to produce the final .tsv matrix __or__ use each step separately, as you see fit
```
#!/bin/bash

for sampleID in $(ls *_1.fastq.gz | sed "s/_1.fastq.gz//g") # adjust the extension if needed

do
    # Step 1: MEGAHIT assembly

    megahit -1 ${sampleID}_1.fastq.gz -2 ${sampleID}_2.fastq.gz -o ${sampleID}_mh-prodigal-output

    # Step 2: Prediction of protein-coding genes and translation into amino acids with Prodigal

    cd ${sampleID}_mh-prodigal-output
    mv final.contigs.fa ${sampleID}.fna # Rename assembly with the correct sampleID identifier
    prodigal -i ${sampleID}.fna -o ${sampleID}.coords.gbk -a ${sampleID}.faa -p meta
    sed -i.save "s/^>/>${sampleID}|/g" ${sampleID}.faa # Modify headers to keep trace of the sampleID in the concatenated .faa file

    # Step 3: Concatenate all single metagenome .faa files into a global .faa file

    cat ${sampleID}.faa >> ../metagenomes_combined.faa
    
    cd ..

done

# Step 4: Clustering sequences with CD-HIT (in this case at 70% identity and 90% of coverage)

cd-hit -i metagenomes_combined.faa -o metagenomes_combined -c 0.70 -aS 0.90 -d 0 -M 0 -T 0 -g 1 -G 0

# Step 5: Transform the CD-HIT .clstr output file to a .tsv matrix using our custom Python script.

python matrixFromCD-Hit-nbcontigs.py metagenomes_combined.clstr > metagenomes_combined.clstr.tsv
```

metagenomes_combined.clstr.tsv
```
ClusterID	SRR123456	SRR234567	SRR345678
Cluster_0	0	0	0
Cluster_1	0	1	0
Cluster_2	1	0	2
Cluster_3	0	0	0
...
Cluster_n	0	0	4
```
