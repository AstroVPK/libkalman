for object in ~/Desktop/Kepler/*/; do
    cd "${object}"llc
    cp * .. 
    cd .. 
    for f in *fits; do 
#echo ${f##*/}
fb=${f##*/}
fb=${fb%.fits}
fb=$fb.dat
#echo $fb
java -jar ~/topcat-full.jar -stilts tcopy ${f##*/} ${fb} ofmt=ascii
    done 
    #rm *fits

done