read -e -p "Path to Kepler data: " -i "~/Documents/Research/Kepler/" keplerPath
cd "${keplerPath}"
for object in ./*/; do
    cd "${object}"llc
    cp * .. 
    cd .. 
    for f in *fits; do 
#echo ${f##*/}
fb=${f##*/}
fb=${fb%.fits}
fb=$fb.dat
#echo $fb
topcat -stilts tcopy ${f##*/} ${fb} ofmt=ascii
    done 
    rm *fits

done