#include "SimpleMOC-kernel_header.h"

// Initialize global flux states to random numbers on device
// Slow, poor use of GPU, but fine since it's just initialization code
__global__ void	init_flux_states( float * flux_states, int N_flux_states, Input I, unsigned long * state)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x; // geometric segment	
	//int threadId = blockId * blockDim.x + threadIdx.x; // energy group

	if(blockId >= N_flux_states)
		return;

	// Assign RNG state
	unsigned long * localState = &state[blockId % I.streams];

	if( threadIdx.x == 0 )
		for( int i = 0; i < I.egroups; i++ )
		{
			LCG_RNG(localState); // update state to next in sequence
			flux_states[blockId +i] = (double) *localState / 2147483647;
		}
}

// Gets I from user and sets defaults
Input set_default_input( void )
{
	Input I;

	I.source_regions = 2250;
	I.course_axial_intervals = 9;
	I.fine_axial_intervals = 5;
	I.segments = 50000000;
	I.egroups = 100;
	I.streams = 10000;

	return I;
}

// Returns a memory esimate (in MB) for the program's primary data structures
double mem_estimate( Input I )
{
	size_t nbytes = 0;

	// Sources Array
	nbytes += I.source_regions * sizeof(Source);

	// Fine Source Data
	long N_fine = I.source_regions * I.fine_axial_intervals * I.egroups;
	nbytes += N_fine * sizeof(float);

	// Fine Flux Data
	nbytes += N_fine * sizeof(float);

	// SigT Data
	long N_sigT = I.source_regions * I.egroups;
	nbytes += N_sigT * sizeof(float);

	// Return MB
	return (double) nbytes / 1024.0 / 1024.0;
}

Source * initialize_sources( Input I, Source_Arrays * SA )
{
	// Source Data Structure Allocation
	Source * sources = (Source *) malloc( I.source_regions * sizeof(Source));

	// Allocate Fine Source Data
	long N_fine = I.source_regions * I.fine_axial_intervals * I.egroups;
	SA->fine_source_arr = (float *) malloc( N_fine * sizeof(float));
	for( int i = 0; i < I.source_regions; i++ )
		sources[i].fine_source_id = i*I.fine_axial_intervals*I.egroups;

	// Allocate Fine Flux Data
	SA->fine_flux_arr = (float *) malloc( N_fine * sizeof(float));
	for( int i = 0; i < I.source_regions; i++ )
		sources[i].fine_flux_id = i*I.fine_axial_intervals*I.egroups;

	// Allocate SigT Data
	long N_sigT = I.source_regions * I.egroups;
	SA->sigT_arr = (float *) malloc( N_sigT * sizeof(float));
	for( int i = 0; i < I.source_regions; i++ )
		sources[i].sigT_id = i * I.egroups;

	// Initialize fine source and flux to random numbers
	for( long i = 0; i < N_fine; i++ )
	{
		SA->fine_source_arr[i] = rand() / RAND_MAX;
		SA->fine_flux_arr[i] = rand() / RAND_MAX;
	}

	// Initialize SigT Values
	for( int i = 0; i < N_sigT; i++ )
		SA->sigT_arr[i] = rand() / RAND_MAX;

	return sources;
}

Source * initialize_device_sources( Input I, Source_Arrays * SA_h, Source_Arrays * SA_d, Source * sources_h )
{
	// Allocate & Copy Fine Source Data
	long N_fine = I.source_regions * I.fine_axial_intervals * I.egroups;
	cudaMalloc((void **) &SA_d->fine_source_arr, N_fine * sizeof(float));
	cudaMemcpy(SA_d->fine_source_arr, SA_h->fine_source_arr, N_fine * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate & Copy Fine Flux Data
	cudaMalloc((void **) &SA_d->fine_flux_arr, N_fine * sizeof(float));
	cudaMemcpy(SA_d->fine_flux_arr, SA_h->fine_flux_arr, N_fine * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate & Copy SigT Data
	long N_sigT = I.source_regions * I.egroups;
	cudaMalloc((void **) &SA_d->sigT_arr, N_sigT * sizeof(float));
	cudaMemcpy(SA_d->sigT_arr, SA_h->sigT_arr, N_sigT * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate & Copy Source Array Data
	Source * sources_d;
	cudaMalloc((void **) &sources_d, I.source_regions * sizeof(Source));
	cudaMemcpy(sources_d, sources_h, I.source_regions * sizeof(Source), cudaMemcpyHostToDevice);

	return sources_d;
}

// Builds a table of exponential values for linear interpolation
Table buildExponentialTable( void )
{
	// define table
	Table table;

	//float precision = 0.01;
	float maxVal = 10.0;	

	// compute number of arry values
	//int N = (int) ( maxVal * sqrt(1.0 / ( 8.0 * precision * 0.01 ) ) );
	int N = 353; 

	// compute spacing
	float dx = maxVal / (float) N;

	// store linear segment information (slope and y-intercept)
	for( int n = 0; n < N; n++ )
	{
		// compute slope and y-intercept for ( 1 - exp(-x) )
		float exponential = exp( - n * dx );
		table.values[ 2*n ] = - exponential;
		table.values[ 2*n + 1 ] = 1 + ( n * dx - 1 ) * exponential;
	}

	// assign data to table
	table.dx = dx;
	table.maxVal = maxVal - table.dx;
	table.N = N;

	return table;
}

void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}
