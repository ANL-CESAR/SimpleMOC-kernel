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

#include <cuda.h>
#include <curand_kernel.h>

// User inputs
typedef struct{
	int source_regions;
	int course_axial_intervals;
	int fine_axial_intervals;
	long segments;
	int egroups;
	int nthreads;
} Input;

// Source Region Structure
typedef struct{
	long fine_flux_id;
	long fine_source_id;
	long sigT_id;
} Source;

// Source Arrays
typedef struct{
	float * fine_flux_arr;
	float * fine_source_arr;
	float * sigT_arr;
} Source_Arrays;


// Table structure for computing exponential
typedef struct{
	float values[706];
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
__global__ void run_kernel( Input I, Source * S,
		Source_Arrays * SA, Table * table, curandState * state,
		float * state_fluxes, int N_state_fluxes)
__device__ void interpolateTable(Table table, float x, float * out)

// init.c
double mem_estimate( Input I );
__global__ void setup_kernel(curandState *state);
Source * initialize_sources( Input I, Source_Arrays * SA );
Source * initialize_device_sources( Input I, Source_Arrays * SA_h, Source_Arrays * SA_d, Source * sources_h );
Table buildExponentialTable( void );
Input set_default_input( void );
SIMD_Vectors aligned_allocate_simd_vectors(Input I);
SIMD_Vectors allocate_simd_vectors(Input I);
double get_time(void);

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
void print_input_summary(Input input);
void read_CLI( int argc, char * argv[], Input * input );
void print_CLI_error(void);

#endif
