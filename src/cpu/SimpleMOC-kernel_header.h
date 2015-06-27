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

#ifdef INTEL
#include<malloc.h>
#endif

#ifdef OPENMP
#include<omp.h>
#endif

#ifdef PAPI
#include<papi.h>
#endif

#ifdef VERIFY
#define float double
#endif

// User inputs
typedef struct{
	int source_2D_regions;
	int source_3D_regions;
	int coarse_axial_intervals;
	int fine_axial_intervals;
	int decomp_assemblies_ax; // Number of subdomains per assembly axially
	long segments;
	int egroups;
	int nthreads;
	size_t nbytes;

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

// kernel.c
unsigned long long run_kernel( Input * I, Source * S, Table * table);
void attenuate_segment( Input * restrict I, Source * restrict S,
		int QSR_id, int FAI_id, float * restrict state_flux,
		SIMD_Vectors * restrict simd_vecs, Table * restrict table); 
float interpolateTable( Table * table, float x);

// init.c
Source * aligned_initialize_sources( Input * I );
Source * initialize_sources( Input * I );
Table * buildExponentialTable( float precision, float maxVal, Input * I );
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
unsigned int hash(unsigned char *str, int nbins);
unsigned int portable_hash( float f );

// papi.c
void papi_serial_init(void);
void counter_init( int *eventset, int *num_papi_events, Input * I );
void counter_stop( int * eventset, int num_papi_events, Input * I );

void print_state_flux( float * flux, int egroups );

#endif
