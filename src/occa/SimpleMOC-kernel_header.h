#ifndef __SimpleMOC_header
#define __SimpleMOC_header

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
#include<sys/time.h>

// User inputs
typedef struct{
  int source_regions;
  int course_axial_intervals;
  int fine_axial_intervals;
  long segments;
  int egroups;
  int nthreads;
  int streams;
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

// Device Source Arrays
typedef struct{
  occaMemory fine_flux_arr;
  occaMemory fine_source_arr;
  occaMemory sigT_arr;
} OCCA_Source_Arrays;

// Table structure for computing exponential
typedef struct{
  float values[706];
  float dx;
  float maxVal;
  int N;
} Table;

// kernel.c

// init.c
double mem_estimate( Input I );
Source * initialize_sources( Input I, Source_Arrays * SA );
occaMemory initialize_occa_sources( Input I, Source_Arrays * SA_h, OCCA_Source_Arrays * SA_d, Source * sources_h, occaDevice device );
Table buildExponentialTable( void );
Input set_default_input( void );

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
void print_input_summary(Input input);
void read_CLI( int argc, char * argv[], Input * input );
void print_CLI_error(void);

#endif
