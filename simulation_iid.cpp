#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<cfloat>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

using namespace std;

#define COMMENT_END_IDENTIFIER ": "
#define SAMPLE_UNIFORM_01 (static_cast<double> (rand())/static_cast<double> (RAND_MAX));
#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

// ################# Parameters for the simulation
vector<double> arrival_batch_size, arrival_prob;
vector<double> connect_rate;
vector< vector<double> > results_ctmb;
int max_buffer, num_queues;
long max_iterations;
int DEBUG = 1;
int POLICY = 0;
int DISTRIBUTION = 0;

// ################# For random number generation using the GNU Scientific Library
const gsl_rng_type *T;
gsl_rng *r;
gsl_ran_discrete_t *g;

void initialize_random_gen(unsigned long s)
{
  srand(s);
  gsl_rng_set(r, s);
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
}

void cleanup_random_gen()
{
  gsl_ran_discrete_free (g);
  gsl_rng_free (r); 
}

// ################# Output routines
template <class T>
void print_vector(vector<T> v, string s)
{
  unsigned int i;
  cout << s << ": (";
  for (i = 0; i < v.size() - 1; i ++)
    cout << v[i] << ",";
  cout << v[i] << ")" << endl;
}

template <class T>
void print_2d_vector(vector<vector<T> > v, string s)
{
  unsigned int i,j;
  cout << s << ": (";
  for (i = 0; i < v.size(); i++){
    for(j = 0; j < v[i].size() - 1; j++)
      cout << v[i][j] << ",";
    cout << v[i][j] << ";";
  }
  cout << ")" << endl;
}

template <class T>
void print_vector_noeol(vector<T> v, string s)
{
  unsigned int i;
  cout << s << ": (";
  for (i = 0; i < v.size() - 1; i ++)
    cout << v[i] << ",";
  cout << v[i] << ") ";
}

template <class T>
void print_cumulativetime_at_queuelength(vector<T> v)
{
  unsigned int i,j;
  for (i = 0; i < v.size(); i++){
    for(j = 0; j < v[i].size(); j++)
      cout << v[i][j] << ",";
    cout << v[i][j] << endl;
  }
}

// ################# Parser for reading the configuration file
inline void rd_line_to_str(istringstream &iss, ifstream &ifile, string &line)
{
  getline(ifile, line);
  line.erase(0, line.find(COMMENT_END_IDENTIFIER) + 2);
  iss.clear();
  iss.str(line);
}

void input_file_parser(char *file_name)
{
  string line; 
  istringstream iss;

  ifstream ifile;
  ifile.open(file_name);

  rd_line_to_str(iss, ifile, line);
  iss >> max_iterations;

  rd_line_to_str(iss, ifile, line);
  iss >> max_buffer;

  rd_line_to_str(iss, ifile, line);
  iss >> num_queues;

  double tb, t;
  for (int i = 0; i < num_queues; i ++) {
    rd_line_to_str(iss, ifile, line);
    iss >> tb >> t;
    arrival_batch_size.push_back(tb);
    arrival_prob.push_back(t);
  }

  for (int i = 0; i < num_queues; i ++) {
    rd_line_to_str(iss, ifile, line);
    iss >> t;
    connect_rate.push_back(t);
  }

  if (DEBUG) {
    cout << max_iterations << " " << max_buffer << " " << num_queues << endl;
    print_vector(arrival_batch_size, "BS");
    print_vector(arrival_prob, "A");
    print_vector(connect_rate, "C");
  }
}

// ################# Schedulers
int weighted_longest_connected_queue(vector<int> q, vector<int> c, int m, int slots_in_service)
{
  int lcq = 0, lcqi = m, tq = 0;
  for (int qi = 0; qi < num_queues; qi ++) {
    if (c[qi] > 0) {
      if (m == qi) {
	tq += slots_in_service;
      }
      if (q[qi] > lcq) {
	lcq = q[qi];
	lcqi = qi;
      }
    }
  }
  return lcqi;
}

int longest_queue(vector<int> q, vector<int> c, int m)
{
  int lcq = 0, lcqi = m;
  for (int qi = 0; qi < num_queues; qi ++) {
    if (q[qi] > lcq) {
      lcq = q[qi];
      lcqi = qi;
    }
  }
  return lcqi;
}  

int longest_connected_queue(vector<int> q, vector<int> c, int m)
{
  int lcq = 0, lcqi = m;
  for (int qi = 0; qi < num_queues; qi ++) {
    if (c[qi] > 0) {
      if (q[qi] > lcq) {
	lcq = q[qi];
	lcqi = qi;
      }
    }
  }
  return lcqi;
}

int exhaustive_service(vector<int> q, vector<int> c, int m)
{
  int lcqi;
  if (q[m] > 0) {
    lcqi = m;
  } else {
    lcqi = longest_connected_queue(q, c, m);
  }
  return lcqi;
}

int exprule(vector<int> q, vector<int> c, int m)
{
  int lcq = 0, lcqi = m;
  int sumq = 0;
  for (int qi = 0; qi < num_queues; qi ++)
    sumq += q[qi];

  sumq = (double) sumq / (double) num_queues;
  lcq = exp((double) q[0] / (1.0 + sqrt((double) sumq)));

  for (int qi = 0; qi < num_queues; qi ++) {
    if (exp((double) q[qi] / (1.0 + sqrt((double) sumq))) > lcq) {
      lcq = exp((double) q[qi] / (1.0 + sqrt((double) sumq)));
      lcqi = qi;
    }
  }
  return lcqi;
}

// ################# Parameters for different distributions are collected in a single structure
struct DistributionParameter {
  double p;
  unsigned long n;
  vector<double> P;
  double lambda;
};

DistributionParameter arrival_parameter;
DistributionParameter connection_parameter;

// ################# IID samplers for obtaining the random number of arrivals in each slot
int sample_binomial(DistributionParameter parameter)
{
  double p = parameter.p;
  unsigned long n = parameter.n;
  int k = gsl_ran_binomial(r,p,n);
  return k;  
}

int sample_general_discrete(DistributionParameter parameter)
{
  vector<double> P = parameter.P;
  size_t K = P.size();
  g = gsl_ran_discrete_preproc (K,P); // see if this is ok!
  size_t d = gsl_ran_discrete (r,g);  
  return d;
}

void sample_bernoulli(DistributionParameter parameter)
{
  double p = parameter.p;
  double sample = SAMPLE_UNIFORM_01;
  if (sample < p) {
    return parameter.n;
  } else {
    return 0;
  }
}

unsigned int sample_truncated_poisson(DistributionParameter parameter)
{
  unsigned int n = parameter.n;
  double lambda = parameter.lambda;
  double *Q = new double[n];
  for(unsigned int i = 0; i < n; i++){
    Q[i] = gsl_ran_poisson_pdf(i,lambda); //poisson pmf
  }   
  return sample_general_discrete( n, Q);
}

unsigned int sample_heavy_tailed_discrete(DistributionParameter parameter)
{
  unsigned int n = parameter.n;
  double alpha = parameter.alpha;
  double Q[n];
  if(alpha<1)
    return -1;
  else
    alpha = (-1)*alpha;
  for(unsigned int i=0;i<n;i++){
    double x = (double)(rand()%10)+1;
    Q[i] = pow(x,alpha);
  }
  return sample_general_discrete(n,Q);
}

// ################# Structure to organize the results
struct Results {
  vector<double> fraction_lost_arrivals;
  vector<double> average_queue_length;
  vector< vector<double> > queue_length_distribution;
};

Results simulation_result;

// ################# The simulation loop
void simulation()
{
  int m = 0; // we are initializing the queue served in the last slot to zero
  vector<int> q(num_queues, 0); // we are initializing all queues to zero; this holds the current queue length
  vector<int> cumulative_arrivals(num_queues, 0); // this holds the total number of arrivals to the queues
  vector<int> lost_arrivals(num_queues, 0); // this holds the arrivals which are lost from the system due to lack of buffer space
  vector<int> cumulative_queuelength(num_queues,0); // this holds the cumulative queue length  
  vector< vector<int> > cumulativetime_at_queuelength(num_queues, vector<int>(max_buffer+1, 0)); // this holds the cumulative time spent by each queue at each queue length *** 
  vector<double> temp(num_queues, 0);

  vector<int> current_arrival(num_queues, 0);
  vector<int> current_connection(num_queues, 0), curr_a(num_queues, 0);
  vector<int> current_service(num_queues, 0);
  
  unsigned int served_queue_index;

  int slots_in_service = 0;
  
  for (long i = 0; i < max_iterations; i ++) {    
    // sample arrivals and connectivity for each queue
    for (int qi = 0; qi < num_queues; qi ++) {

      if (DISTRIBUTION == 1) {
	current_arrival[qi] = sample_binomial(arrival_parameter);
      } else if (DISTRIBUTION == 2) {
	current_arrival[qi] = sample_truncated_poisson(arrival_parameter);
      } else if (DISTRIBUTION == 3) {
	current_arrival[qi] = sample_general_discrete(arrival_parameter);
      } else if (DISTRIBUTION == 4) {
	current_arrival[qi] = sample_bernoulli(arrival_parameter);
      } else if (DISTRIBUTION == 5) {
	current_arrival[qi] = sample_heavy_tailed_discrete(arrival_parameter);
      }
      current_connection[qi] = sample_bernoulli(connection_parameter);
    }

    if (POLICY == 1) {
      served_queue_index = longest_connected_queue(q, current_connection, m);
    } else if (POLICY == 2) {
      served_queue_index = exhaustive_service(q, current_connection, m);
    } else if (POLICY == 3) {
      served_queue_index = longest_queue(q, current_connection, m);
    } else if (POLICY == 4) {
      served_queue_index = weighted_longest_connected_queue(q, current_connection, m, slots_in_service);
    } else if (POLICY == 5) {
      served_queue_index = exprule(q, current_connection, m);
    }

    if (m == served_queue_index)
      slots_in_service ++;
    else
      slots_in_service = 0;
    
    current_service[m] = 0;
    current_service[served_queue_index] = 1;

    if (DEBUG) {
      cout << i << " - " << "M: " << m << " ";
      print_vector_noeol(q, "Q");
      print_vector_noeol(curr_a, "A");
      print_vector_noeol(current_connection, "C");
      print_vector_noeol(current_service, "S");
    }

    for (int qi = 0; qi < num_queues; qi ++) {
      cumulative_arrivals[qi] += current_arrival[qi];

      for(int j = 0; j <= max_buffer; j++)
	cumulativetime_at_queuelength[qi][j] += (q[qi] == j);
      
      current_service[qi] = current_connection[qi] * (m == qi) * current_service[qi];
      
      lost_arrivals[qi] += MAX(MAX(q[qi] - current_service[qi], 0) + current_arrival[qi] - max_buffer,0);
      
      q[qi] = MIN( MAX(q[qi] - current_service[qi], 0) + current_arrival[qi], max_buffer);
      
      cumulative_queuelength[qi] += q[qi];
    }

    m = served_queue_index;

    if (DEBUG) {
      print_vector_noeol(lost_arrivals, "LA");
      print_2d_vector(cumulativetime_at_queuelength, "@B");
    }
  }
  
  for (int qi = 0; qi < num_queues; qi ++){
    simulation_results.fraction_lost_arrivals.push_back( (double) lost_arrivals[qi] / (double) cumulative_arrivals[qi]);
    simulation_results.average_queue_length.push_back( (double) cumulative_queuelength[qi] / (double) max_iterations);
    
    for(int j = 0; j <= max_buffer; j++) {
      temp[j] = (double) cumulativetime_at_queuelength[qi][j] / (double) max_iterations;
    }

    simulation_results.queue_length_distribution.push_back(temp);
  }
}

// Main function for setup of arguments
int main(int argc, char *argv[])
{
  if (argc != 12) {
    cout << "Usage: ./simulation_iid_ac_2q <buffersize> <batchsize queue 1> <batchsize queue 2> <arrival prob 1> <arrival prob 2> <conn. prob 1> <conn. prob 2> <iterations> <debug> <policy> <distribution>" << endl;
    return -1;
  }
  num_queues = 2;
  max_buffer = atoi(argv[1]);
  arrival_batch_size.push_back(atoi(argv[2]));
  arrival_batch_size.push_back(atoi(argv[3]));
  arrival_prob.push_back(atof(argv[4]));
  arrival_prob.push_back(atof(argv[5]));
  connect_rate.push_back(atof(argv[6]));
  connect_rate.push_back(atof(argv[7]));
  max_iterations = atoi(argv[8]);
  DEBUG = atoi(argv[9]);
  POLICY = atoi(argv[10]);
  DISTRIBUTION = atoi(argv[11]);
  
  initialize_random_gen(time(NULL));
  
  results = simulation();

  //  cout << max_buffer << "," << arrival_batch_size[0] << "," << arrival_batch_size[1] << "," << arrival_prob[0] * arrival_batch_size[0] << "," << arrival_prob[1] * arrival_batch_size[1] << "," << connect_rate[0] << "," << connect_rate[1] << "," << results[0] << "," << results[2] << "," << results[1] << "," << results[3] << endl;

  // cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;

  cleanup_random_gen();
  return 0;
}
