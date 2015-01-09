#include "SimpleMOC-kernel_header.h"

__global__ void setup_kernel(curandState *state)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x
	/* Each thread gets same seed, a different sequence 
	 *        number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

// Initialize global flux states to random numbers on device
__global__ void	init_flux_states( float * flux_states, int N_flux_states, Input I, curandState * state)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x; // geometric segment	
	int threadId = blockId * blockDim.x + threadIdx.x; // energy group

	curandState localState = state[threadId];

	flux_states[threadID] = curand_uniform(&localState);
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

	// OpenMP Locks
	#ifdef OPENMP
	nbytes += I.source_regions * I.course_axial_intervals * sizeof(omp_lock_t);
	#endif

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

	// Allocate Locks
	#ifdef OPENMP
	SA->locks_arr = init_locks(I);
	for( int i = 0; i < I.source_regions; i++)
		sources[i].locks_id = i * I.course_axial_intervals;
	#endif

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
	cudaMemcpy(SA_d->fine_source_arr, SA_h->fine_source_arr,
			N_fine * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate & Copy Fine Flux Data
	cudaMalloc((void **) &SA_d->fine_flux_arr, N_fine * sizeof(float));
	cudaMemcpy(SA_d->fine_flux_arr, SA_h->fine_flux_arr,
			N_fine * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate & Copy SigT Data
	long N_sigT = I.source_regions * I.egroups;
	cudaMalloc((void **) &SA_d->sitT_arr, N_sigT * sizeof(float));
	cudaMemcpy(SA_d->sigT_arr, SA_h->sigT_arr,
			N_sigT * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate & Copy Source Array Data
	Source * sources_d;
	cudaMalloc((void **) &sources_d, I.source_regions * sizeof(Source));
	cudaMemcpy(sources_d, sources_h, I.source_regions * sizeof(Source),
			cuaMemcpyHostToDevice);

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

#ifdef INTEL
SIMD_Vectors aligned_allocate_simd_vectors(Input I)
{
	SIMD_Vectors A;
	A.q0 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.q1 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.q2 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.sigT = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.tau = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.sigT2 =(float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.expVal =(float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.reuse = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.flux_integral = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.tally = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.t1 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.t2 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.t3 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	A.t4 = (float *) _mm_malloc(I.egroups * sizeof(float), 64);
	return A;
}
#endif

SIMD_Vectors allocate_simd_vectors(Input I)
{
	SIMD_Vectors A;
	float * ptr = (float * ) malloc( I.egroups * 14 * sizeof(float));
	A.q0 = ptr;
	ptr += I.egroups;
	A.q1 = ptr;
	ptr += I.egroups;
	A.q2 = ptr;
	ptr += I.egroups;
	A.sigT = ptr;
	ptr += I.egroups;
	A.tau = ptr;
	ptr += I.egroups;
	A.sigT2 = ptr;
	ptr += I.egroups;
	A.expVal = ptr;
	ptr += I.egroups;
	A.reuse = ptr;
	ptr += I.egroups;
	A.flux_integral = ptr;
	ptr += I.egroups;
	A.tally = ptr;
	ptr += I.egroups;
	A.t1 = ptr;
	ptr += I.egroups;
	A.t2 = ptr;
	ptr += I.egroups;
	A.t3 = ptr;
	ptr += I.egroups;
	A.t4 = ptr;

	return A;
}

#ifdef OPENMP
// Intialized OpenMP Source Region Locks
omp_lock_t * init_locks( Input I )
{
	// Allocate locks array
	long n_locks = I.source_regions * I.course_axial_intervals; 
	omp_lock_t * locks = (omp_lock_t *) malloc( n_locks* sizeof(omp_lock_t));

	// Initialize locks array
	for( long i = 0; i < n_locks; i++ )
		omp_init_lock(&locks[i]);

	return locks;
}	
#endif

// Timer function. Depends on if compiled with MPI, openmp, or vanilla
double get_time(void)
{
	#ifdef OPENMP
	return omp_get_wtime();
	#endif

	time_t time;
	time = clock();

	return (double) time / (double) CLOCKS_PER_SEC;
}
