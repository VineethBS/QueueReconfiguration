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

// Parameters for the simulation
vector<double> arrival_batch_size, arrival_prob;
vector<double> connect_rate;
vector< vector<double> > results_ctmb;
int max_buffer, num_queues;
long max_iterations;
int DEBUG = 1;
int POLICY = 0;
int DISTRIBUTION = 0;
const gsl_rng_type *T;
gsl_rng *r;
gsl_ran_discrete_t *g;

void initialize_rndg()
{
  srand(time(NULL));
}

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
void print_cumtime_maxbuffer(vector<T> v)
{
  unsigned int i,j;
  for (i = 0; i < v.size(); i++){
    for(j = 0; j < v[i].size(); j++)
      cout << v[i][j] << ",";
    cout << v[i][j] << endl;
  }
}

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

int sample_binomial(double p, unsigned int n)
{
  int k = gsl_ran_binomial(r,p,n);
  return k;  
}

int sample_general_discrete(size_t K, const double * P)
{
  g = gsl_ran_discrete_preproc (K,P);
  size_t d = gsl_ran_discrete (r,g);  
  return d;
}

void sample_bernoulli( vector<double> a, vector<int> *b, int m)
{
  double sample = SAMPLE_UNIFORM_01;
  if (sample < a[m]) {
    (*b)[m] = 1;
  } else {
    (*b)[m] = 0;
  }
}

unsigned int sample_truncated_poisson(unsigned int n, double mu)
{
  double Q[n];
  // double P =  gsl_cdf_poisson_P(n,mu);
  for(unsigned int i=0;i<n;i++){
    Q[i] = gsl_ran_poisson_pdf(i,mu); //poisson pmf
  }
   
  return sample_general_discrete(n,Q); // n is total number of discrete events
}

unsigned int sample_heavy_tailed_discrete(unsigned int n,double alpha)
{
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

vector<double> simulation()
{
  int m = 0; // we are initializing the queue served in the last slot to zero
  vector<int> q(num_queues, 0); // we are initializing all queues to zero
  vector<int> lost_arrivals(num_queues, 0), total_arrivals(num_queues, 0), total_qlen(num_queues,0);
  vector<vector<int> > cumtime_maxbuffer(num_queues, vector<int>(max_buffer+1, 0));
  vector< vector<double> > results_ctmb; 
  vector<double> temp;
  vector<int> curr_arr(num_queues, 0);
  vector<int> curr_conn(num_queues, 0), curr_a(num_queues, 0);
  vector<int> curr_ser(num_queues, 0);
  // temp.reserve(51);
  // double sample;
  int lcqi;
  vector<double> results;
  int slots_in_service = 0;
  double p = 0.7; 
  unsigned long s = 0; //seed
  unsigned int N = 10; //number of outcomes
  double P[5] = {0.5,0.45,0.85,0.9,0.1}; //discrete events
  double mu = 2, alpha = 1.2;
  
  gsl_rng_set (r,s);
  
  for (long i = 0; i < max_iterations; i ++) {    
    // sample arrivals and connectivity for each queue
    for (int qi = 0; qi < num_queues; qi ++) {

      if (DISTRIBUTION == 1) {
	curr_arr[qi] = sample_binomial(p,N);
      } else if (DISTRIBUTION == 2) {
	curr_arr[qi] = sample_truncated_poisson(N,mu);
      } else if (DISTRIBUTION == 3) {
	curr_arr[qi] = sample_general_discrete(N,P);
      } else if (DISTRIBUTION == 4) {
	sample_bernoulli(arrival_prob,&curr_a,qi);
	curr_arr[qi] = curr_a[qi] * arrival_batch_size[qi];
      } else if (DISTRIBUTION == 5) {
	curr_arr[qi] = sample_heavy_tailed_discrete(N,alpha);
      }
      sample_bernoulli(connect_rate,&curr_conn,qi);
    }

    if (POLICY == 1) {
      lcqi = longest_connected_queue(q, curr_conn, m);
    } else if (POLICY == 2) {
      lcqi = exhaustive_service(q, curr_conn, m);
    } else if (POLICY == 3) {
      lcqi = longest_queue(q, curr_conn, m);
    } else if (POLICY == 4) {
      lcqi = weighted_longest_connected_queue(q, curr_conn, m, slots_in_service);
    } else if (POLICY == 5) {
      lcqi = exprule(q, curr_conn, m);
    }

    if (m == lcqi)
      slots_in_service ++;
    else
      slots_in_service = 0;
    
    curr_ser[m] = 0;
    curr_ser[lcqi] = 1;

    if (DEBUG) {
      cout << i << " - " << "M: " << m << " ";
      print_vector_noeol(q, "Q");
      print_vector_noeol(curr_a, "A");
      print_vector_noeol(curr_conn, "C");
      print_vector_noeol(curr_ser, "S");
    }

    for (int qi = 0; qi < num_queues; qi ++) {
      //total_arrivals[qi] += curr_a[qi] * arrival_batch_size[qi];
      total_arrivals[qi] += curr_arr[qi];

      for(int j = 0; j <= max_buffer; j++)
	cumtime_maxbuffer[qi][j] += (q[qi] == j);
      
      curr_ser[qi] = curr_conn[qi] * (m == qi) * curr_ser[qi]; // modifications due to connectivity
      lost_arrivals[qi] += MAX(MAX(q[qi] - curr_ser[qi], 0) + curr_arr[qi] - max_buffer,0);
      q[qi] = MIN( MAX(q[qi] - curr_ser[qi], 0) + curr_arr[qi], max_buffer);
      total_qlen[qi] += q[qi];
    }

    m = lcqi;

    if (DEBUG) {
      print_vector_noeol(lost_arrivals, "LA");
      print_2d_vector(cumtime_maxbuffer, "@B");
    }
  }
  for (int qi = 0; qi < num_queues; qi ++){
    results.push_back( (double) lost_arrivals[qi] / (double) total_arrivals[qi]);
    //results.push_back( (double) cumtime_maxbuffer[qi] / (double) max_iterations);
    results.push_back( (double) total_qlen[qi] / (double) max_iterations);

    for(int j = 0; j <= max_buffer; j++)
      temp.push_back( (double) cumtime_maxbuffer[qi][j] / (double) max_iterations);

    results_ctmb.push_back(temp);
    //  cout << results_ctmb.size() << endl;
    //  cout << results_ctmb[qi].size() << endl;
    temp.clear();  
  }
  int j;
  for(j = 0;j< max_buffer;j++)
    cout << j << ",";
  cout << j << endl;
  print_cumtime_maxbuffer(results_ctmb);
  return results;
}

int main(int argc, char *argv[])
{
  vector<double> results;

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
  
  initialize_rndg();
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  
  results = simulation();

  //  cout << max_buffer << "," << arrival_batch_size[0] << "," << arrival_batch_size[1] << "," << arrival_prob[0] * arrival_batch_size[0] << "," << arrival_prob[1] * arrival_batch_size[1] << "," << connect_rate[0] << "," << connect_rate[1] << "," << results[0] << "," << results[2] << "," << results[1] << "," << results[3] << endl;

  // cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
  
  gsl_ran_discrete_free (g);
  gsl_rng_free (r); 
  return 0;
}
