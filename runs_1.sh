#python3 main.py -nf_prior None -nf_vardistr None -data goodreads -gpu 1 -batch_size_test 87 -n_IS 1 -num_epoches 5 -metric recall
#python3 main.py -nf_prior IAF -num_flows_prior 5 -nf_vardistr NAF -num_flows_vardistr 2 -data goodreads -gpu 1 -batch_size_test 87 -n_IS 1 -num_epoches 5 -metric recall
#python3 main.py -nf_prior RNVP -num_flows_prior 5 -nf_vardistr NAF -num_flows_vardistr 2 -data goodreads -gpu 1 -batch_size_test 87 -n_IS 1 -num_epoches 5 -metric recall

python3 main.py -nf_prior None -nf_vardistr None -data mnist -gpu 1 -batch_size_test 100 -n_IS 1
python3 main.py -nf_prior None -nf_vardistr NAF -num_flows_vardistr 5 -data mnist -gpu 1 -batch_size_test 100 -n_IS 1
python3 main.py -nf_prior IAF -num_flows_prior 5 -nf_vardistr None -data mnist -gpu 1 -batch_size_test 100 -n_IS 1
python3 main.py -nf_prior RNVP -num_flows_prior 5 -nf_vardistr None -data mnist -gpu 1 -batch_size_test 100 -n_IS 1
python3 main.py -nf_prior IAF -num_flows_prior 5 -nf_vardistr NAF -num_flows_vardistr 2 -data mnist -gpu 1 -batch_size_test 100 -n_IS 1
python3 main.py -nf_prior RNVP -num_flows_prior 5 -nf_vardistr NAF -num_flows_vardistr 2 -data mnist -gpu 1 -batch_size_test 100 -n_IS 1