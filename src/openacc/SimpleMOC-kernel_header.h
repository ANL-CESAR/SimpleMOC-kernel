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
#include<malloc.h>

#ifdef OPENMP
#include<omp.h>
#endif

#ifdef PAPI
#include<papi.h>
#endif

// User inputs
typedef struct{
	int source_regions;
	int fine_axial_intervals;
	long segments;
	int egroups;
	int nthreads;
  int n_state_fluxes;
} Input;

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

// kernel.c
void run_kernel( 
    int       source_regions,
    int       fine_axial_intervals,
    long      segments,
    int       egroups,
    int       nthreads,
    int       n_state_fluxes,
    float     (* restrict fine_flux_arr)[fine_axial_intervals][egroups], 
    float     (* restrict fine_source_arr)[fine_axial_intervals][egroups],
    float     (* restrict sigT_arr)[egroups],
    float     (* restrict state_flux_arr)[egroups],
    unsigned  (* restrict randIdx)[3]
    );
// init.c
//Source * aligned_initialize_sources( Input * I );
void initialize_sources( 
    int source_regions, 
    int fine_axial_intervals, 
    int egroups,
    float (*fine_flux_arr)[fine_axial_intervals][egroups],
    float (*fine_source_arr)[fine_axial_intervals][egroups],
    float (*sigT)[egroups]
    );
void initialize_state_flux( 
    int n_state_fluxes, 
    int egroups, 
    float (* state_flux_arr)[egroups] 
    );
void initialize_randIdx( int segments, unsigned (*randIdx)[3]);
Table * buildExponentialTable( float precision, float maxVal );
Input * set_default_input( void );
SIMD_Vectors aligned_allocate_simd_vectors(Input * I);
SIMD_Vectors allocate_simd_vectors(Input * I);
double get_time(void);
#ifdef OPENMP
omp_lock_t * init_locks( Input * I );
#endif

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
void print_input_summary(Input * input);
void read_CLI( int argc, char * argv[], Input * input );
void print_CLI_error(void);
void read_input_file( Input * I, char * fname);

// papi.c
void papi_serial_init(void);
void counter_init( int *eventset, int *num_papi_events, Input * I );
void counter_stop( int * eventset, int num_papi_events, Input * I );

#endif
