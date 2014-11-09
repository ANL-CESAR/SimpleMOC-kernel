#ifndef __SimpleMOC_header
#define __SimpleMOC_header

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>
#include<stdbool.h>
#include<limits.h>
#include<assert.h>
#include<pthread.h>
#include<unistd.h>

#ifdef OPENMP
#include<omp.h>
#endif

#ifdef PAPI
#include<papi.h>
#endif

// User inputs
typedef struct{
	int source_regions;
	int course_axial_intervals;
	int fine_axial_intervals;
	long segments;
	int egroups;

	#ifdef OPENMP
	int nthreads;
	#endif

    #ifdef PAPI
	int papi_event_set;
    // String for command line PAPI event
    char event_name[PAPI_MAX_STR_LEN]; 
    // Array to accumulate PAPI counts across all threads
    long long *vals_accum;
    #endif
} Input;



// Source Region Structure
typedef struct{
	float * fine_flux;
	float * fine_source;
	float * sigT;
	#ifdef OPENMP
	omp_lock_t * locks;
	#endif
} Source;

// Table structure for computing exponential
typedef struct{
	float * values;
	float dx;
	float maxVal;
	int N;
} Table;

// Local SIMD Vector Arrays
typedef struct{
	float * q0;
	float * q1;
	float * q2;
	float * sigT;
	float * tau;
	float * sigT2;
	float * expVal;
	float * reuse;
	float * flux_integral;
	float * tally;
	float * t1;
	float * t2;
	float * t3;
	float * t4;
} SIMD_Vectors;


// init.c
Input set_default_input( void );
void set_small_input( Input * I );
Params build_tracks( Input I );
CommGrid init_mpi_grid( Input I );
void calculate_derived_inputs( Input * I );
#ifdef OPENMP
omp_lock_t * init_locks( Input I );
#endif

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
void print_input_summary(Input input);
void read_CLI( int argc, char * argv[], Input * input );
void print_CLI_error(void);
void read_input_file( Input * I, char * fname);

// tracks.c
Track2D * generate_2D_tracks( Input input, size_t * nbytes );
void generate_2D_segments( Input input, Track2D * tracks,
	   	size_t * nbytes );
void free_2D_tracks( Track2D * tracks );
Track *** generate_tracks(Input input, Track2D * tracks_2D, size_t * nbytes);
void free_tracks( Track *** tracks );
long segments_per_2D_track_distribution( Input I );
float * generate_polar_angles( Input I );

// utils.c
float urand(void);
float nrand(float mean, float sigma);
float pairwise_sum(float * vector, long size);
Table buildExponentialTable( float precision, float maxVal );
float interpolateTable( Table table, float x);
double get_time(void);
size_t est_mem_usage( Input I );
double time_per_intersection( Input I, double time );

// source.c
Source * initialize_sources( Input I, size_t * nbytes );
void free_sources( Input I, Source * sources );

// solver.c
void transport_sweep( Params params, Input I );
int get_pos_interval( float z, float dz);
int get_neg_interval( float z, float dz);
int get_alt_neg_interval( float z, float dz);
void attenuate_fluxes( Track * track, Source * QSR, Input * I, 
		Params * params, float ds, float mu, float az_weight, AttenuateVars * A ); 
void attenuate_FSR_fluxes( Track * track, Source * FSR, Input * I,
		Params * params, float ds, float mu, float az_weight, AttenuateVars * A );
void alt_attenuate_fluxes( Track * track, Source * FSR, Input * I,
		Params * params, float ds, float mu, float az_weight );
void renormalize_flux( Params params, Input I, CommGrid grid );
float update_sources( Params params, Input I, float keff );
float compute_keff( Params params, Input I, CommGrid grid);

// test.c
void gen_norm_pts(float mean, float sigma, int n_pts);
void print_Input_struct( Input I );

// papi.c
void papi_serial_init(void);
void counter_init( int *eventset, int *num_papi_events, Input I );
void counter_stop( int * eventset, int num_papi_events, Input * I );

#endif
