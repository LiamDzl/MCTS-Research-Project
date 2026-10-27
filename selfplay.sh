for i in $(seq 1 12); do
    python -u selfplay.py $i &
done
wait