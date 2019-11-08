#Restricted Boltzmann Machine Launcher
#Copyright 2020 Denis Rothman MIT License. READ LICENSE.
import RBM as rp

#Create feature files
f=open("features.tsv","w+")
f.close

g=("viewer_name"+"\t"+"primary_emotion"+"\t"+"secondary_emotion"+"\n")
with open("labels.tsv", "w") as f:
    f.write(g)

#Run the RBM feature detection program over v viewers
print("RBM start")
vn=12001
c=0
for v in range (0,vn):
    rp.main()
    c+=1
    if(c==1000):print(v+1);c=0;

print("RBM over")

