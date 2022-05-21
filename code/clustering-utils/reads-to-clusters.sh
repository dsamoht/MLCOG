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