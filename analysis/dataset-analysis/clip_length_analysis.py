from collections import Counter

csv_path = '/data/jongmin/projects/dataset/kinetics.csv'

with open(csv_path, 'r') as f:
    lines = f.readlines() 
frame_length = []
for line in lines[1:]: 
    attrs = line.split(',')
    length = int(attrs[4]) - int(attrs[3]) 
    frame_length.append(length) 


print(Counter(frame_length))
