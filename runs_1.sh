#python3 main.py -nf_prior None -nf_vardistr None -data mnist -gpu 1 -batch_size_test 100 -n_IS 1
python3 main.py -nf_prior NAF -nf_vardistr NAF -data mnist -gpu 1 -batch_size_test 100 -n_IS 1