import glob
import os
images = glob.glob('./phototest/*')
print(images)

n=1
for i in images:
    os.rename(i, f'./phototest/female{n}.jpg')
    n+=1