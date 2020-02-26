#python3 main.py -nf_prior None -nf_vardistr None -data goodreads -gpu 0
#
#python3 main.py -nf_prior NAF -nf_vardistr None -data goodreads -gpu 0
#python3 main.py -nf_prior NAF -num_nafs_prior 3 -nf_vardistr None -data goodreads -gpu 0
#python3 main.py -nf_prior NAF -num_nafs_prior 5 -nf_vardistr None -data goodreads -gpu 0
#
#python3 main.py -nf_prior None -nf_vardistr NAF -data goodreads -gpu 0
#python3 main.py -nf_prior None -nf_vardistr NAF -num_nafs_vardistr 3 -data goodreads -gpu 0
#python3 main.py -nf_prior None -nf_vardistr NAF -num_nafs_vardistr 5 -data goodreads -gpu 0
#
#python3 main.py -nf_prior NAF -nf_vardistr NAF -data goodreads -gpu 0
#python3 main.py -nf_prior NAF -num_nafs_prior 3 -nf_vardistr NAF -num_nafs_vardistr 3 -data goodreads -gpu 0
python3 main.py -nf_prior NAF -num_nafs_prior 5 -nf_vardistr NAF -num_nafs_vardistr 5 -data goodreads -gpu 0