for i in `seq 0 49`
do
  ./build/helloworld /mnt/poplin/tmp/ammae/data/ 12 $i
done

sh ./ifttt.sh program_finished. learning owari
