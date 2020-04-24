401
#!/bin/bash

module purge
module load python/3.7

python3 #FNAME#


for i in {1...500}
do
submit.sh exp11$i.sh
done