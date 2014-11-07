#include<SimpleMOC-kernel_header.h>

void run_kernel( Input * I, Params * P)
{
	// Enter Parallel Region
	#pragma omp parallel default(none) shared(I, P)
	{
		// Create Thread Local Random Seed
		unsigned int seed = time(NULL) * (thread+1);

		// Allocate Thread Local SIMD Vectors
		SIMD_vectors simd_vectors = allocate_simd_vectors(I);

		// Allocate Thread Local Flux Vector
		float * state_flux = (float *) malloc( I->n_egroups * sizeof(float));
		for( int i = 0; i < I->n_egroups; i++ )
			state_flux[i] = r_rand(&seed) / RAND_MAX;

		// Enter OMP For Loop over Segments
		#pragma omp for schedule(dynamic)
		for( long i = 0; i < I.n_segments; i++ )
		{
			// Pick Random QSR
			int QSR_id = rand_r(&seed) % I->n_source_regions;

			// Attenuate Segment
			attenuate_segment( I, P, QSR_id, state_flux, local_vectors);
		}
	}
}

void attenuate_segment( Input * I, Params * P, float * state_flux,
		int QSR_id, Vectors * vectors) 
{
	Input I = *I_in;
	Params params = *params_in;

	// unload attenuate vars
	float * restrict q0 = A->q0;
	float *  restrict q1 = A->q1;
	float *  restrict q2 = A->q2;
	float *  restrict sigT = A->sigT;
	float *  restrict tau = A->tau;
	float *  restrict sigT2 = A->sigT2;
	float *  restrict expVal = A->expVal;
	float *  restrict reuse = A->reuse;
	float *  restrict flux_integral = A->flux_integral;
	float *  restrict tally = A->tally;
	float *  restrict t1 = A->t1;
	float *  restrict t2 = A->t2;
	float *  restrict t3 = A->t3;
	float *  restrict t4 = A->t4;

	// compute fine axial interval spacing
	//float dz = I.height / (I.fai * I.decomp_assemblies_ax * I.cai);
	float dz = 0.1f;

	// compute z height in cell
	float zin = 0.2f - dz * 
		( (int)( 0.3f / dz ) + 0.5f );

	// compute fine axial region ID
	int fine_id = (int) ( 0.3f / dz ) % 5;

	// compute weight (azimuthal * polar)
	// NOTE: real app would also have volume weight component
	float weight = 0.1f * 0.2f;
	float mu2 = 0.3f;

	// load fine source region flux vector
	float * FSR_flux = QSR -> fine_flux[fine_id];

	if( fine_id == 0 )
	{
		// cycle over energy groups
		#ifdef INTEL
		#pragma simd
		#elif defined IBM
		#pragma simd_level(10)
		#endif
		for( int g = 0; g < I.n_egroups; g++)
		{
			// load neighboring sources
			float y2 = QSR->fine_source[fine_id][g];
			float y3 = QSR->fine_source[fine_id+1][g];

			// do linear "fitting"
			float c0 = y2;
			float c1 = (y3 - y2) / dz;

			// calculate q0, q1, q2
			q0[g] = c0 + c1*zin;
			q1[g] = c1;
			q2[g] = 0;
		}
	}
	else if ( fine_id == I.fai - 1 )
	{
		// cycle over energy groups
		#ifdef INTEL
		#pragma simd
		#elif defined IBM
		#pragma simd_level(10)
		#endif
		for( int g = 0; g < I.n_egroups; g++)
		{
			// load neighboring sources
			float y1 = QSR->fine_source[fine_id-1][g];
			float y2 = QSR->fine_source[fine_id][g];

			// do linear "fitting"
			float c0 = y2;
			float c1 = (y2 - y1) / dz;

			// calculate q0, q1, q2
			q0[g] = c0 + c1*zin;
			q1[g] = c1;
			q2[g] = 0;
		}
	}
	else
	{
		// cycle over energy groups
		#ifdef INTEL
		#pragma simd
		#elif defined IBM
		#pragma simd_level(10)
		#endif
		for( int g = 0; g < I.n_egroups; g++)
		{
			// load neighboring sources
			float y1 = QSR->fine_source[fine_id-1][g];
			float y2 = QSR->fine_source[fine_id][g];
			float y3 = QSR->fine_source[fine_id+1][g];

			// do quadratic "fitting"
			float c0 = y2;
			float c1 = (y1 - y3) / (2.f*dz);
			float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

			// calculate q0, q1, q2
			q0[g] = c0 + c1*zin + c2*zin*zin;
			q1[g] = c1 + 2.f*c2*zin;
			q2[g] = c2;
		}
	}


	// cycle over energy groups
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		// load total cross section
		sigT[g] = QSR->sigT[g];

		// calculate common values for efficiency
		tau[g] = sigT[g] * ds;
		sigT2[g] = sigT[g] * sigT[g];
	}

	// cycle over energy groups
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
		expVal[g] = interpolateTable( params.expTable, tau[g] );  

	// Flux Integral

	// Re-used Term
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		reuse[g] = tau[g] * (tau[g] - 2.f) + 2.f * expVal[g] 
			/ (sigT[g] * sigT2[g]); 
	}

	//#pragma vector nontemporal
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		// add contribution to new source flux
		flux_integral[g] = (q0[g] * tau[g] + (sigT[g] * track->psi[g] - q0[g])
				* expVal[g]) / sigT2[g] + q1[g] * mu * reuse[g] + q2[g] * mu2 
			* (tau[g] * (tau[g] * (tau[g] - 3.f) + 6.f) - 6.f * expVal[g]) 
			/ (3.f * sigT2[g] * sigT2[g]);
	}

	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		// Prepare tally
		tally[g] = weight * flux_integral[g];
	}

	#ifdef OPENMP
	omp_set_lock(QSR->locks + fine_id);
	#endif

	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		FSR_flux[g] += tally[g];
	}

	#ifdef OPENMP
	omp_unset_lock(QSR->locks + fine_id);
	#endif

	// Term 1
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		t1[g] = q0[g] * expVal[g] / sigT[g];  
	}
	// Term 2
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		t2[g] = q1[g] * mu * (tau[g] - expVal[g]) / sigT2[g]; 
	}
	// Term 3
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		t3[g] =	q2[g] * mu2 * reuse[g];
	}
	// Term 4
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		t4[g] = track->psi[g] * (1.f - expVal[g]);
	}
	// Total psi
	#ifdef INTEL
	#pragma simd
	#elif defined IBM
	#pragma simd_level(10)
	#endif
	for( int g = 0; g < I.n_egroups; g++)
	{
		track->psi[g] = t1[g] + t2[g] + t3[g] + t4[g];
	}
}	

/* Interpolates a formed exponential table to compute ( 1- exp(-x) )
 *  at the desired x value */
float interpolateTable( Table table, float x)
{
	// check to ensure value is in domain
	if( x > table.maxVal )
		return 1.0f;
	else
	{
		int interval = (int) ( x / table.dx + 0.5f * table.dx );
		/*
		   if( interval >= table.N || interval < 0)
		   {
		   printf( "Interval = %d\n", interval);
		   printf( "N = %d\n", table.N);
		   printf( "x = %f\n", x);
		   printf( "dx = %f\n", table.dx);
		   exit(1);
		   }
		   */
		float slope = table.values[ 2 * interval ];
		float intercept = table.values[ 2 * interval + 1 ];
		float val = slope * x + intercept;
		return val;
	}
}
