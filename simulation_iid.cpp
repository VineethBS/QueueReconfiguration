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

#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

// ################# Indices used to extract parameters from the command line
#define IND_SEED 1
#define IND_MAX_BUFFER 2
#define IND_NUM_QUEUES 3
#define IND_MAX_ITERATIONS 4
#define IND_DEBUG 5
#define IND_POLICY 6
#define IND_DISTRIBUTION 7
#define IND_OUTPUTTYPE 8
#define IND_FIXEDPARAMETERS_END 9
#define NUM_PARAMS_PERQUEUE 4
#define INDOFFSET_ARRIVAL_p 0
#define INDOFFSET_ARRIVAL_n 1
#define INDOFFSET_ARRIVAL_lambda 2
#define INDOFFSET_CONNECTION_p 3

// ################# Parameters for the simulation
int seed;
long max_buffer;
long max_iterations;
int num_queues;
int debug = 0;
int policy = 0;
int distribution = 0;
int output_type = 0;

// ################# Parameters for different distributions are collected in a single structure
struct DistributionParameter {
  double p;
  long n;
  vector<double> P;
  double lambda;
};

vector<DistributionParameter> arrival_parameter;
vector<DistributionParameter> connection_parameter;

// ################# For random number generation using the GNU Scientific Library
const gsl_rng_type *T;
gsl_rng *r;

void initialize_random_gen( long s)
{
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set(r, s);
}

void cleanup_random_gen()
{
  gsl_rng_free (r); 
}

// ################# Useful vector operations
template <class T>
vector<T> multiply_vector(vector<T> v, double n)
{
  for (unsigned int i = 0; i < v.size(); i ++)
    v[i] = v[i] * n;
  return v;
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
  int i,j;
  for (i = 0; i < v.size(); i++){
    for(j = 0; j < v[i].size(); j++)
      cout << v[i][j] << ",";
    cout << v[i][j] << endl;
  }
}

// ################# Schedulers
int weighted_longest_connected_queue(vector< long> q, vector< long> c,  int m,  int slots_in_service)
{
  long lcq = 0;
  int lcqi = m;
  int tq = 0;
  
  for ( int qi = 0; qi < num_queues; qi ++) {
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

int longest_queue(vector< long> q, vector< long> c,  int m)
{
  long lq = 0;
  int lqi = m;
  
  for ( int qi = 0; qi < num_queues; qi ++) {
    if (q[qi] > lq) {
      lq = q[qi];
      lqi = qi;
    }
  }
  return lqi;
}  

int longest_connected_queue(vector< long> q, vector< long> c,  int m)
{
  long lcq = 0;
  int lcqi = m;
  for ( int qi = 0; qi < num_queues; qi ++) {
    if (c[qi] > 0) {
      if (q[qi] > lcq) {
	lcq = q[qi];
	lcqi = qi;
      }
    }
  }
  return lcqi;
}

int exhaustive_service(vector< long> q, vector< long> c,  int m)
{
  int server_index;
  if (q[m] > 0) {
    server_index = m;
  } else {
    server_index = longest_connected_queue(q, c, m);
  }
  return server_index;
}

int exprule(vector< long> q, vector< long> c,  int m)
{
  double metric = 0;
  int server_index = m;
  long sumq = 0;
  for ( int qi = 0; qi < num_queues; qi ++)
    sumq += q[qi];

  sumq = (double) sumq / (double) num_queues;
  metric = exp((double) q[0] / (1.0 + sqrt((double) sumq)));

  for ( int qi =0; qi < num_queues; qi ++) {
    if (exp((double) q[qi] / (1.0 + sqrt((double) sumq))) > metric) {
      metric = exp((double) q[qi] / (1.0 + sqrt((double) sumq)));
      server_index = qi;
    }
  }
  return server_index;
}

int VFMW(vector< long> q,  int m)
{
  double metric = 0;
  int server_index = m;
  
  for ( int qi = 0; qi < num_queues; qi ++) { 
    if (q[qi] * connection_parameter[qi].p > metric) {
      metric = q[qi] * connection_parameter[qi].p;
      server_index = qi;
    }
  }
  return server_index;
}

long frame_length_VFMW(vector< long> q)
{
  int total_queue_length = 0;
  for (unsigned int i = 0; i < q.size(); i ++)
    total_queue_length += q[i];
  long frame_length = ( long) (1 + sqrt((double) total_queue_length));
  return frame_length;
}

				
// ################# IID samplers for obtaining the random number of arrivals in each slot
int sample_binomial(DistributionParameter parameter)
{
  double p = parameter.p;
  long n = parameter.n;
  return gsl_ran_binomial(r,p,n);  
}

int sample_general_discrete(DistributionParameter parameter)
{
  unsigned int i;
  vector<double> P = parameter.P;
  // convert P to a cumulative DF
  for (i = 1; i < P.size(); i ++) {
    P[i] = P[i] + P[i - 1];
  }
  double uniform01_sample = gsl_ran_flat(r, 0, 1);
  for (i = 0; i < P.size(); i ++) {
    if (uniform01_sample <= P[i])
      break;
  }
  return i;
}

int sample_bernoulli(DistributionParameter parameter)
{
  double p = parameter.p;
  return gsl_ran_binomial(r,p,1);
}

int sample_truncated_poisson(DistributionParameter parameter)
{
  // TODO: Optimize this code so that the distribution is only found once
  int n = parameter.n;
  double lambda = parameter.lambda;
  vector<double> Q(n, 0);
  double total_probability = 0;
  
  for( int i = 0; i < n; i++){
    Q[i] = gsl_ran_poisson_pdf(i,lambda); //poisson pmf
    total_probability += Q[i];
  }

  Q = multiply_vector(Q, 1.0/total_probability);

  DistributionParameter temp;
  temp.n = n;
  temp.lambda = 0;
  temp.p = 0;
  temp.P = Q;
  return sample_general_discrete(temp);
}

int sample_heavy_tailed_discrete(DistributionParameter parameter)
{
  int n = parameter.n;
  double alpha = parameter.lambda;
  double total_probability = 0;
  
  vector<double> Q(n, 0);
  
  if ( alpha < 1)
    return -1;
  else
    alpha = (-1)*alpha;
  
  for(  int i=0; i < n; i++) {
    Q[i] = pow(i, alpha);
    total_probability += Q[i];
  }

  Q = multiply_vector(Q, 1.0/total_probability);

  DistributionParameter temp;
  temp.n = n;
  temp.lambda = 0;
  temp.p = 0;
  temp.P = Q;
  return sample_general_discrete(temp);  
}

// ################# Structure to organize the results
struct Results {
  vector<double> fraction_lost_arrivals;
  vector<double> average_queue_length;
  vector< vector<double> > queue_length_distribution;
};

Results simulation_results;

// ################# Function to print the results
// output_type decides what should be printed, if it is 1 or 2 then a header row should be printed
// if it is 3 or 4 then header row is not printed, if it is 4 then queue length distribution should be printed
void print_results(int output_type)
{
  if ((output_type == 1) || (output_type == 2)) {
    cout << "Seed,MaxBufferSize,NumQueues,MaxIterations,Policy,Distribution";
    for ( int q = 0; q < num_queues; q ++) {
      cout << ",Queue_" << q << "_Arrivalp,Queue_" << q << "_Arrivaln,Queue_" << q << "Arrival_L,Queue_" << q <<"_Connectionp";
    }
    for ( int q = 0; q < num_queues; q ++) {
      cout << ",Queue_" << q << "_FractionLostArrivals,Queue_" << q << "_AverageQueueLength";
    }
    cout << endl;
  } else {
    cout << seed << "," << max_buffer << "," << num_queues << "," << max_iterations << "," << policy << "," << distribution;
    for ( int q = 0; q < num_queues; q ++) {
      cout << arrival_parameter[q].p << "," << arrival_parameter[q].n << "," << arrival_parameter[q].lambda << "," << connection_parameter[q].p;
    }
    for ( int q = 0; q < num_queues; q ++) {
      cout << simulation_results.fraction_lost_arrivals[q] << "," << simulation_results.average_queue_length[q];
    }
    cout << endl;
  }
}

// ################# The simulation loop
void simulation()
{
  if (debug >= 1) {
    cout << "#### Start of Simulation ####" << endl;
  }
  
  int m = 0; // we are initializing the queue served in the last slot to zero
  vector<long> q(num_queues, 0); // we are initializing all queues to zero; this holds the current queue length
  vector<long> cumulative_arrivals(num_queues, 0); // this holds the total number of arrivals to the queues
  vector<long> lost_arrivals(num_queues, 0); // this holds the arrivals which are lost from the system due to lack of buffer space
  vector<long> cumulative_queuelength(num_queues,0); // this holds the cumulative queue length  
  vector< vector<long> > cumulativetime_at_queuelength(num_queues, vector<long>(max_buffer + 1, 0)); // this holds the cumulative time spent by each queue at each queue length
  vector<double> temp(max_buffer + 1, 0);

  vector<long> current_arrival(num_queues, 0);
  vector<long> current_connection(num_queues, 0), curr_a(num_queues, 0);
  vector<long> current_service(num_queues, 0);
  
  long served_queue_index;

  long slots_in_service = 0;
  
  long frame_length = 0;
  long frame_boundary = frame_length;
  
  for ( long i = 0; i < max_iterations; i ++) {    

    for ( int qi = 0; qi < num_queues; qi ++) {
      // sample arrivals for each queue according to specified distribution
      if (distribution == 1) {
	current_arrival[qi] = sample_binomial(arrival_parameter[qi]);
      } else if (distribution == 2) {
       	current_arrival[qi] = sample_truncated_poisson(arrival_parameter[qi]);
      } else if (distribution == 3) {
      	current_arrival[qi] = sample_general_discrete(arrival_parameter[qi]);
      } else if (distribution == 4) {
	current_arrival[qi] = sample_bernoulli(arrival_parameter[qi]);
      } else if (distribution == 5) {
      	current_arrival[qi] = sample_heavy_tailed_discrete(arrival_parameter[qi]);
      }
      // sample connectivity for each queue - connectivity is bernoulli
      current_connection[qi] = sample_bernoulli(connection_parameter[qi]);
    }

    if (policy == 1) {
      served_queue_index = longest_connected_queue(q, current_connection, m);
    } else if (policy == 2) {
      served_queue_index = exhaustive_service(q, current_connection, m);
    } else if (policy == 3) {
      served_queue_index = longest_queue(q, current_connection, m);
    } else if (policy == 4) {
      served_queue_index = weighted_longest_connected_queue(q, current_connection, m, slots_in_service);
    } else if (policy == 5) {
      served_queue_index = exprule(q, current_connection, m);
    } else if (policy == 6) {
      if (i == frame_boundary) {
	served_queue_index = VFMW(q, m);
	frame_length = frame_length_VFMW(q);
	frame_boundary = frame_boundary + frame_length;
      }
    }

    if (m == served_queue_index)
      slots_in_service ++;
    else
      slots_in_service = 0;
    
    current_service[m] = 0;
    current_service[served_queue_index] = 1;

    if (debug >= 1) {
      cout << i << " - " << "M: " << m << " ";
      print_vector_noeol(q, "Q");
      print_vector_noeol(current_arrival, "A");
      print_vector_noeol(current_connection, "C");
      print_vector(current_service, "S");
    }

    for (int qi = 0; qi < num_queues; qi ++) {
      cumulative_arrivals[qi] += current_arrival[qi];

      for(long j = 0; j <= max_buffer; j++)
       	cumulativetime_at_queuelength[qi][j] += (q[qi] == j);
      
      current_service[qi] = current_connection[qi] * (m == qi) * current_service[qi];
      
      lost_arrivals[qi] += MAX(MAX(q[qi] - current_service[qi], 0) + current_arrival[qi] - max_buffer,0);
      
      q[qi] = MIN( MAX(q[qi] - current_service[qi], 0) + current_arrival[qi], max_buffer);
      
      cumulative_queuelength[qi] += q[qi];
    }

    m = served_queue_index;

    if (debug >= 2) {
      print_vector_noeol(lost_arrivals, "LA");
      print_2d_vector(cumulativetime_at_queuelength, "@B");
    }
  }
  
  for ( int qi = 0; qi < num_queues; qi ++){
    simulation_results.fraction_lost_arrivals.push_back( (double) lost_arrivals[qi] / (double) cumulative_arrivals[qi]);
    simulation_results.average_queue_length.push_back( (double) cumulative_queuelength[qi] / (double) max_iterations);
    
    for( long j = 0; j <= max_buffer; j++) {
      temp[j] = (double) cumulativetime_at_queuelength[qi][j] / (double) max_iterations;
    }

    simulation_results.queue_length_distribution.push_back(temp);
  }

  if (debug >= 1) {
    cout << "#### End of Simulation ####" << endl;
  }
}

// Main function for setup of arguments
int main(int argc, char *argv[])
{
  seed = atoi(argv[ IND_SEED ]);
  max_buffer = atoi(argv[ IND_MAX_BUFFER ]);
  num_queues = atoi(argv[ IND_NUM_QUEUES ]);
  max_iterations = atoi(argv[ IND_MAX_ITERATIONS ]);
  debug = atoi(argv[ IND_DEBUG ]);
  policy = atoi(argv[ IND_POLICY ]);
  distribution = atoi(argv[ IND_DISTRIBUTION ]);
  output_type = atoi(argv[ IND_OUTPUTTYPE ]);
  DistributionParameter arrival_temp;
  DistributionParameter connection_temp;


  for ( int q = 0; q < num_queues; q ++) {
    arrival_temp.p = atof(argv[ IND_FIXEDPARAMETERS_END + q * NUM_PARAMS_PERQUEUE + INDOFFSET_ARRIVAL_p ]);
    arrival_temp.n = atoi(argv[ IND_FIXEDPARAMETERS_END + q * NUM_PARAMS_PERQUEUE + INDOFFSET_ARRIVAL_n ]);
    arrival_temp.lambda = atof(argv[ IND_FIXEDPARAMETERS_END + q * NUM_PARAMS_PERQUEUE + INDOFFSET_ARRIVAL_lambda ]);
    connection_temp.p = atof(argv[ IND_FIXEDPARAMETERS_END + q * NUM_PARAMS_PERQUEUE + INDOFFSET_CONNECTION_p ]);
    connection_temp.n = 0;
    connection_temp.lambda = 0;

    arrival_parameter.push_back(arrival_temp);
    connection_parameter.push_back(connection_temp);
  }

  // Debug the inputs
  if (debug >= 1) {
    cout << "#### Inputs read from the command line are #### " << endl;
    cout << "Seed " << seed << endl;
    cout << "Max buffer size " << max_buffer << endl;
    cout << "Number of queues " << num_queues << endl;
    cout << "Max Iterations " << max_iterations << endl;
    cout << "Policy " << policy << endl;
    cout << "Distribution " << distribution << endl;

    for ( int q = 0; q < num_queues; q ++) {
      cout << "Queue 1 | Arrival.p " << arrival_parameter[q].p << " | Arrival.n " << arrival_parameter[q].n << " | Arrival L " << arrival_parameter[q].lambda << " | Connection p " << connection_parameter[q].p << endl;
    }
    cout << "#### End of Inputs #### " << endl;
    
  }
  
  initialize_random_gen(seed);  	
  simulation();
  print_results();
  cleanup_random_gen();  

  return 0;
}
