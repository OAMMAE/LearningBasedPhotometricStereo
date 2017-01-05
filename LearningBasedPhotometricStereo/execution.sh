for i in `seq $1 $2`
do
  ./build/helloworld /mnt/poplin/tmp/ammae/data/ 8 $i
  sh ./ifttt.sh $3 learning $i
done

