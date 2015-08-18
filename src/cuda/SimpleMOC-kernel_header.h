#ifndef __SimpleMOC_header
#define __SimpleMOC_header

#include <curand_kernel.h>
#include <cuda.h>
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
#include<assert.h>

#define CUDA_ERROR_CHECK
 
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

// CUDA Error Handling Macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

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
	int streams;
	int seg_per_thread;
	size_t nbytes;
} Input;

// Source Region Structure
typedef struct{
	long fine_flux_id;
	long fine_source_id;
	long sigT_id;
} Source;

// Source Arrays
typedef struct{
	double * fine_flux_arr;
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

// kernel.c
__global__ void run_kernel( Input I, Source *  S,
		Source_Arrays SA, Table *  table, curandState *  state,
		float *  state_fluxes, int N_state_fluxes);
__device__ void interpolateTable(Table *  table, float x, float *  out);
__device__ double double_atomicAdd(double* address, double val);

// init.c
double mem_estimate( Input I );
__global__ void setup_kernel(curandState *state, Input I);
__global__ void	init_flux_states( float * flux_states, int N_flux_states, Input I, curandState * state);
Source * initialize_sources( Input I, Source_Arrays * SA );
Source * initialize_device_sources( Input I, Source_Arrays * SA_h, Source_Arrays * SA_d, Source * sources_h );
Table buildExponentialTable( void );
Input set_default_input( void );
void __cudaCheckError( const char *file, const int line );

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
void print_input_summary(Input input);
void read_CLI( int argc, char * argv[], Input * input );
void print_CLI_error(void);

#endif
