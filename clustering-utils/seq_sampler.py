""" 1st argument: fasta file
   2nd argument: file: headers of interest(1 header per line)
   stdout: filtered fasta file
"""
import sys


id_to_seq = {}

with open(sys.argv[1], "r") as fasta_file:
    for line in fasta_file:
        if line.startswith(">"):
            current_id = line
            id_to_seq[current_id] = ""
        if line.startswith(">") == False:
            id_to_seq[current_id] += line.strip()

with open(sys.argv[2], "r") as header_file:
    for line in header_file:
        try:
            print(line,  end="")
            print(id_to_seq[line])
        except KeyError:
            print("Identifier Error")
            print(""""{}" is not in the list of headers""".format(line))
