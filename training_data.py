import random

print('lets go!!')
def create_input_data(pos, neg):

    data=list()
    with open(pos,'r') as f:
        lines=f.readlines()
        print(lines[0])
        print(lines[0].split()+[1])
        for i in lines:
            data.append((i.split(),1))
    with open(neg,'r') as f:
        lines=f.readlines()
        print(lines[0])
        print(lines[0].split()+[0])
        for i in lines:
            data.append((i.split(),0))

    random.shuffle(data)

    return data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
            
