python run_inference.py &> beam_search_multinomial_sampling1.txt
echo "done 0"
python run_inference1.py &> greedy_search.txt
echo "done 1"
python run_inference2.py &> multinomial_sampling.txt
echo "done 2"

python run_inference3.py &> contrastive_search.txt
echo "done 3"