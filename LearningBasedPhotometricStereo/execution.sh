for i in `seq $1 $2`
do
  ./build/helloworld /mnt/poplin/tmp/ammae/data/ 4 $i
  sh ./ifttt.sh program_finished. learning $i
done

