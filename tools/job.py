import glob
import os

# ---------------------------------------------------------------------------------------------
path1 = "D:/Downloads/traffic_signs_dataset/classification/train"
path2 = "D:/Downloads/traffic_signs_dataset/classification/test"
li1 = [c[0] for c in os.walk(path1)]
li1 = li1[1:]
li1 = [int(c.split("\\")[1]) for c in li1]
li1.sort()

i1 = 0
i2 = 0
c = 0
for l in li1:
    imgs1 = glob.glob1(path1 + "/" + str(l), "*")
    imgs2 = glob.glob1(path2 + "/" + str(l), "*")
    i1 += len(imgs1)
    i2 += len(imgs2)
    print("Class: " + str(l) + " Train: " + str(len(imgs1)) + " Test: " + str(len(imgs2)))
    c += 1

print(i1, " ", i2, " ", c)
