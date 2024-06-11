import os
path = '/home/raghav/Documents/BTP/IMAGES/DATASET/'
files = os.listdir(path)
i =1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path,'d'+str(i)+'.jpg'))
    i = i+1

print i