for i in $1 $2
do
  ./build/helloworld /mnt/poplin/tmp/ammae/data/ 8 $i $3
  sh ./ifttt.sh $4 learning $i
done

