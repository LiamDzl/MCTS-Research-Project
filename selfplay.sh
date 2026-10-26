for i in $(seq 1 12); do
    python selfplay.py $i &
done
wait