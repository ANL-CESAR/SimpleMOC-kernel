#include"SimpleMOC_header.h"

Source * initialize_sources( Input * I )
{
	// Allocate Space for Source Array
	Source * sources = (Source *) malloc(); 
}

// Builds a table of exponential values for linear interpolation
Table buildExponentialTable( float precision, float maxVal )
{
    // define table
    Table table;

    // compute number of arry values
    int N = (int) ( maxVal * sqrt(1.0 / ( 8.0 * precision * 0.01 ) ) );

    // compute spacing
    float dx = maxVal / (float) N;

    // allocate an array to store information
    float * tableVals = malloc( (2 * N ) * sizeof(float) );

    // store linear segment information (slope and y-intercept)
    for( int n = 0; n < N; n++ )
    {
        // compute slope and y-intercept for ( 1 - exp(-x) )
        float exponential = exp( - n * dx );
        tableVals[ 2*n ] = - exponential;
        tableVals[ 2*n + 1 ] = 1 + ( n * dx - 1 ) * exponential;
    }

    // assign data to table
    table.dx = dx;
    table.values = tableVals;
    table.maxVal = maxVal - table.dx;
    table.N = N;

    return table;
}

// Gets I from user and sets defaults
Input set_default_input( void )
{
	Input I;

	I.source_regions = 2250;

	I.course_axial_intervals = 9;

	I.fine_axial_intervals = 5;
	
	I.segments = 10000000;
	
	I.egroups = 100;

	#ifdef PAPI
	I.papi_event_set = 6;
	#endif
	
	#ifdef OPENMP
	I.nthreads = omp_get_max_threads();
	#endif

	return I;
}

SIMD_Vectors allocate_simd_vectors(Input * I)
{
	SIMD_Vectors A;
	float * ptr = (float * ) malloc( I->n_egroups * 14 * sizeof(float));
	A.q0 = ptr;
	ptr += I.n_egroups;
	A.q1 = ptr;
	ptr += I.n_egroups;
	A.q2 = ptr;
	ptr += I.n_egroups;
	A.sigT = ptr;
	ptr += I.n_egroups;
	A.tau = ptr;
	ptr += I.n_egroups;
	A.sigT2 = ptr;
	ptr += I.n_egroups;
	A.expVal = ptr;
	ptr += I.n_egroups;
	A.reuse = ptr;
	ptr += I.n_egroups;
	A.flux_integral = ptr;
	ptr += I.n_egroups;
	A.tally = ptr;
	ptr += I.n_egroups;
	A.t1 = ptr;
	ptr += I.n_egroups;
	A.t2 = ptr;
	ptr += I.n_egroups;
	A.t3 = ptr;
	ptr += I.n_egroups;
	A.t4 = ptr;
	
	return A;
}

#ifdef OPENMP
// Intialized OpenMP Source Region Locks
omp_lock_t * init_locks( Input I )
{
	// Allocate locks array
	long n_locks = I.n_source_regions_per_node * I.cai; 
	omp_lock_t * locks = (omp_lock_t *) malloc( n_locks* sizeof(omp_lock_t));

	// Initialize locks array
	for( long i = 0; i < n_locks; i++ )
		omp_init_lock(&locks[i]);

	return locks;
}	
#endif


