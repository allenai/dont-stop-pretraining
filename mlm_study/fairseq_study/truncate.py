import sys 

file_path = sys.argv[1]
input_file = open(file_path, 'r').readlines()
output_file = open(file_path.replace('.bpe', '.truncate.bpe'), 'w')

for line in input_file:
	if line != '\n':
		line_split = line.split(' ')
		if len(line_split) > 510: 
			output_file.write(' '.join(line_split[:510]) + '\n')
		else:
			output_file.write(line)
	else: 
		output_file.write(line) 
