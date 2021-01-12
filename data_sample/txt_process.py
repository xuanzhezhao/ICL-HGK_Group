
with open("B/0.txt", "r") as f:
    lines = f.readlines()
i=1
with open("B/b.txt", "w") as f:
    for line in lines:
        if i%62!=1 and i%62!=2:
            f.write(line)
        i=i+1
