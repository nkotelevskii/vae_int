python3 main.py -nf_prior None -data mnist -gpu 0
python3 main.py -nf_prior NAF -data mnist -gpu 0
python3 main.py -n_samples 10 -batch_size_train 100 -nf_prior None -data mnist -gpu 0
python3 main.py -n_samples 10 -batch_size_train 100 -nf_prior NAF -num_nafs 5 -data mnist -gpu 0