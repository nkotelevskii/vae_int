python3 main.py -nf_prior None -nf_vardistr None -data big_dataset -gpu 0 -metric recall -csv_path ./data/big_dataset/librate.csv  -n_IS 1
#python3 main.py -nf_prior NAF -nf_vardistr None -data goodreads -gpu 0
#python3 main.py -nf_prior NAF -num_flows_prior 3 -nf_vardistr None -data goodreads -gpu 0
#python3 main.py -nf_prior NAF -num_flows_prior 5 -nf_vardistr None -data goodreads -gpu 0
#
#python3 main.py -nf_prior None -nf_vardistr NAF -data goodreads -gpu 0
#python3 main.py -nf_prior None -nf_vardistr NAF -num_flows_vardistr 3 -data goodreads -gpu 0
#python3 main.py -nf_prior None -nf_vardistr NAF -num_flows_vardistr 5 -data goodreads -gpu 0
#
#python3 main.py -nf_prior NAF -nf_vardistr NAF -data goodreads -gpu 0
#python3 main.py -nf_prior NAF -num_flows_prior 3 -nf_vardistr NAF -num_flows_vardistr 3 -data goodreads -gpu 0  -metric recall
#python3 main.py -nf_prior NAF -num_flows_prior 5 -nf_vardistr NAF -num_flows_vardistr 5 -data goodreads -gpu 0