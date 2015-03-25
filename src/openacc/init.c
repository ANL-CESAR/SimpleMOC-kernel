#include "SimpleMOC-kernel_header.h"

// Gets I from user and sets defaults
Input * set_default_input( void )
{
	Input * I = (Input *) malloc(sizeof(Input));

	I->source_regions = 2250;
	I->fine_axial_intervals = 5;
	I->segments = 50000000;
	I->egroups = 128;
  I->n_state_fluxes = 10000;

	#ifdef PAPI
	I->papi_event_set = 0;
	#endif

	#ifdef OPENMP
	I->nthreads = omp_get_max_threads();
	#endif

	return I;
}

void initialize_sources( 
    int source_regions, 
    int fine_axial_intervals, 
    int egroups,
    float (* restrict *fine_flux_arr)[fine_axial_intervals][egroups],
    float (* restrict * fine_source_arr)[fine_axial_intervals][egroups],
    float (* restrict * sigT_arr)[egroups]
    )
{

	// Allocate Fine Source Data
  *fine_source_arr = (float (* restrict)[fine_axial_intervals][egroups])
    malloc( source_regions * fine_axial_intervals * egroups * sizeof(float));

	// Allocate Fine Flux Data
  *fine_flux_arr = (float (* restrict)[fine_axial_intervals][egroups]) 
    malloc( source_regions * fine_axial_intervals * egroups * sizeof(float));

	// Allocate SigT
  *sigT_arr = (float (* restrict)[egroups]) 
    malloc( source_regions * egroups * sizeof(float));

	// Initialize fine source and flux to random numbers
	for( int i = 0; i < source_regions; i++ )
		for( int j = 0; j < fine_axial_intervals; j++ )
			for( int k = 0; k < egroups; k++ )
			{
				(*fine_flux_arr)[i][j][k] = (float) rand() / RAND_MAX;
				(*fine_source_arr)[i][j][k] = (float) rand() / RAND_MAX;
			}

	// Initialize SigT Values
	for( int i = 0; i < source_regions; i++ )
		for( int j = 0; j < egroups; j++ )
			(*sigT_arr)[i][j] = (float) rand() / RAND_MAX;

	return;
}

void initialize_state_flux( 
    int n_state_fluxes, 
    int egroups, 
    float (* restrict * state_flux_arr)[egroups] 
    )
{
  *state_flux_arr = (float (* restrict)[egroups]) 
    malloc(n_state_fluxes * egroups * sizeof(float));

  for (int i = 0; i < n_state_fluxes; i++)
    for (int j = 0; j < egroups; j++)
      (*state_flux_arr)[i][j] = (float) rand() / RAND_MAX;
}


void initialize_randIdx( 
    int segments, 
    unsigned (* restrict * randIdx)[3] )
{
  *randIdx = (unsigned (* restrict)[3]) 
    malloc(segments * 3 * sizeof(unsigned));

  for (int i = 0; i < segments; i++)
    for (int j = 0; j < 3;  j++)
      (*randIdx)[i][j] = (unsigned) rand();

  return;
}


// Builds a table of exponential values for linear interpolation
Table * buildExponentialTable( float precision, float maxVal )
{
	// define table
	Table * table = (Table *) malloc(sizeof(Table));

	// compute number of arry values
	int N = (int) ( maxVal * sqrt(1.0 / ( 8.0 * precision * 0.01 ) ) );

	// compute spacing
	float dx = maxVal / (float) N;

	// allocate an array to store information
	#ifdef INTEL
	float * tableVals = _mm_malloc( (2 * N ) * sizeof(float), 64 );
	#else
	float * tableVals = malloc( (2 * N ) * sizeof(float) );
	#endif

	// store linear segment information (slope and y-intercept)
	for( int n = 0; n < N; n++ )
	{
		// compute slope and y-intercept for ( 1 - exp(-x) )
		float exponential = exp( - n * dx );
		tableVals[ 2*n ] = - exponential;
		tableVals[ 2*n + 1 ] = 1 + ( n * dx - 1 ) * exponential;
	}

	// assign data to table
	table->dx = dx;
	table->values = tableVals;
	table->maxVal = maxVal - table->dx;
	table->N = N;

	return table;
}

// #ifdef INTEL
// SIMD_Vectors aligned_allocate_simd_vectors(int egroups)
// {
// 	SIMD_Vectors A;
// 	A.q0 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.q1 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.q2 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.sigT = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.tau = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.sigT2 =(float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.expVal =(float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.reuse = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.flux_integral = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.tally = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.t1 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.t2 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.t3 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	A.t4 = (float *) _mm_malloc(egroups * sizeof(float), 64);
// 	return A;
// }
// #endif

// SIMD_Vectors allocate_simd_vectors(int egroups)
// {
// 	SIMD_Vectors A;
// 	float * ptr = (float * ) malloc(egroups * 14 * sizeof(float));
// 	A.q0 = ptr;
// 	ptr += egroups;
// 	A.q1 = ptr;
// 	ptr += egroups;
// 	A.q2 = ptr;
// 	ptr += egroups;
// 	A.sigT = ptr;
// 	ptr += egroups;
// 	A.tau = ptr;
// 	ptr += egroups;
// 	A.sigT2 = ptr;
// 	ptr += egroups;
// 	A.expVal = ptr;
// 	ptr += egroups;
// 	A.reuse = ptr;
// 	ptr += egroups;
// 	A.flux_integral = ptr;
// 	ptr += egroups;
// 	A.tally = ptr;
// 	ptr += egroups;
// 	A.t1 = ptr;
// 	ptr += egroups;
// 	A.t2 = ptr;
// 	ptr += egroups;
// 	A.t3 = ptr;
// 	ptr += egroups;
// 	A.t4 = ptr;
// 
// 	return A;
// }
// 
// #ifdef OPENMP
// // Intialized OpenMP Source Region Locks
// omp_lock_t * init_locks( Input * I )
// {
// 	// Allocate locks array
// 	long n_locks = I->source_regions * I->fine_axial_intervals; 
// 	omp_lock_t * locks = (omp_lock_t *) malloc( n_locks* sizeof(omp_lock_t));
// 
// 	// Initialize locks array
// 	for( long i = 0; i < n_locks; i++ )
// 		omp_init_lock(&locks[i]);
// 
// 	return locks;
// }	
// #endif

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
