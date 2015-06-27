#include "SimpleMOC-kernel_header.h"

unsigned long long run_kernel( Input * I, Source * S, Table * table)
{
	unsigned long long vhash = 0;

	// Enter Parallel Region
	#pragma omp parallel default(none) shared(I, S, table, vhash)
	{
		#ifdef OPENMP
		int thread = omp_get_thread_num();
		#else
		int thread = 0;
		#endif

		// Create Thread Local Random Seed
		unsigned int seed = time(NULL) * (thread+1);

		// Allocate Thread Local SIMD Vectors (align if using intel compiler)
		#ifdef INTEL
		SIMD_Vectors simd_vecs = aligned_allocate_simd_vectors(I);
		float * state_flux = (float *) _mm_malloc(
				I->egroups * sizeof(float), 64);
		#else
		SIMD_Vectors simd_vecs = allocate_simd_vectors(I);
		float * state_flux = (float *) malloc(
				I->egroups * sizeof(float));
		#endif

		// Allocate Thread Local Flux Vector
		for( int i = 0; i < I->egroups; i++ )
		{
			state_flux[i] = rand_r(&seed) / RAND_MAX;

			#ifdef VERIFY
			state_flux[i] = 1.0;
			#endif
		}

		// Initialize PAPI Counters (if enabled)
		#ifdef PAPI
		int eventset = PAPI_NULL;
		int num_papi_events;
		#pragma omp critical
		{
			counter_init(&eventset, &num_papi_events, I);
		}
		#endif

		// Enter OMP For Loop over Segments
		#pragma omp for schedule(dynamic,100)
		for( long i = 0; i < I->segments; i++ )
		{
			// Pick Random QSR
			int QSR_id = rand_r(&seed) % I->source_3D_regions;

			// Pick Random Fine Axial Interval
			int FAI_id = rand_r(&seed) % I->fine_axial_intervals;

			// If verification, set inputs deterministically based
			// on segment ID
			#ifdef VERIFY
			QSR_id = i % I->source_3D_regions;
			FAI_id = i % I->fine_axial_intervals;
			#endif

			// Attenuate Segment
			attenuate_segment( I, S, QSR_id, FAI_id, state_flux,
					&simd_vecs, table);

			// If verification, add resulting state flux to hash
			// Then, reset state_flux to 1
			#ifdef VERIFY
			unsigned long long local_hash = 0;
			for( int j = 0; j < I->egroups; j++ )
			{
				/*
				char line[256];
				sprintf(line, "%.5lf", state_flux[j]);
				local_hash += hash(line, 10000);
				*/
				local_hash += portable_hash(state_flux[j]);
				state_flux[j] = 1.0;
			}
			#pragma omp atomic
			vhash += local_hash;
			#endif
		}

		// Stop PAPI Counters
		#ifdef PAPI
		if( thread == 0 )
		{
			printf("\n");
			border_print();
			center_print("PAPI COUNTER RESULTS", 79);
			border_print();
			printf("Count          \tSmybol      \tDescription\n");
		}
		{
			#pragma omp barrier
		}
		counter_stop(&eventset, num_papi_events, I);
		#endif
	}

	return vhash;
}

void attenuate_segment( Input * restrict I, Source * restrict S,
		int QSR_id, int FAI_id, float * restrict state_flux,
		SIMD_Vectors * restrict simd_vecs, Table * restrict table) 
{
	// Unload local vector vectors
	float * restrict q0 =            simd_vecs->q0;
	float * restrict q1 =            simd_vecs->q1;
	float * restrict q2 =            simd_vecs->q2;
	float * restrict sigT =          simd_vecs->sigT;
	float * restrict tau =           simd_vecs->tau;
	float * restrict sigT2 =         simd_vecs->sigT2;
	float * restrict expVal =        simd_vecs->expVal;
	float * restrict reuse =         simd_vecs->reuse;
	float * restrict flux_integral = simd_vecs->flux_integral;
	float * restrict tally =         simd_vecs->tally;
	float * restrict t1 =            simd_vecs->t1;
	float * restrict t2 =            simd_vecs->t2;
	float * restrict t3 =            simd_vecs->t3;
	float * restrict t4 =            simd_vecs->t4;

	// Some placeholder constants - In the full app some of these are
	// calculated based off position in geometry. This treatment
	// shaves off a few FLOPS, but is not significant compared to the
	// rest of the function.
	const float dz = 0.1f;
	const float zin = 0.3f; 
	const float weight = 0.5f;
	const float mu = 0.9f;
	const float mu2 = 0.3f;
	const float ds = 0.7f;

	const int egroups = I->egroups;

	// load fine source region flux vector
	float * FSR_flux = &S[QSR_id].fine_flux[FAI_id * egroups];

	if( FAI_id == 0 )
	{
		float * f2 = &S[QSR_id].fine_source[FAI_id*egroups]; 
		float * f3 = &S[QSR_id].fine_source[(FAI_id+1)*egroups]; 
		// cycle over energy groups
		#ifdef INTEL
		#pragma vector
		#elif defined IBM
		#pragma vector_level(10)
		#endif
		for( int g = 0; g < egroups; g++)
		{
			// load neighboring sources
			const float y2 = f2[g];
			const float y3 = f3[g];

			// do linear "fitting"
			const float c0 = y2;
			const float c1 = (y3 - y2) / dz;

			// calculate q0, q1, q2
			q0[g] = c0 + c1*zin;
			q1[g] = c1;
			q2[g] = 0;
		}
	}
	else if ( FAI_id == I->fine_axial_intervals - 1 )
	{
		float * f1 = &S[QSR_id].fine_source[(FAI_id-1)*egroups]; 
		float * f2 = &S[QSR_id].fine_source[FAI_id*egroups]; 
		// cycle over energy groups
		#ifdef INTEL
		#pragma vector
		#elif defined IBM
		#pragma vector_level(10)
		#endif
		for( int g = 0; g < egroups; g++)
		{
			// load neighboring sources
			const float y1 = f1[g];
			const float y2 = f2[g];

			// do linear "fitting"
			const float c0 = y2;
			const float c1 = (y2 - y1) / dz;

			// calculate q0, q1, q2
			q0[g] = c0 + c1*zin;
			q1[g] = c1;
			q2[g] = 0;
		}
	}
	else
	{
		float * f1 = &S[QSR_id].fine_source[(FAI_id-1)*egroups]; 
		float * f2 = &S[QSR_id].fine_source[FAI_id*egroups]; 
		float * f3 = &S[QSR_id].fine_source[(FAI_id+1)*egroups]; 
		// cycle over energy groups
		#ifdef INTEL
		#pragma vector
		#elif defined IBM
		#pragma vector_level(10)
		#endif
		for( int g = 0; g < egroups; g++)
		{
			// load neighboring sources
			const float y1 = f1[g]; 
			const float y2 = f2[g];
			const float y3 = f3[g];

			// do quadratic "fitting"
			const float c0 = y2;
			const float c1 = (y1 - y3) / (2.f*dz);
			const float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

			// calculate q0, q1, q2
			q0[g] = c0 + c1*zin + c2*zin*zin;
			q1[g] = c1 + 2.f*c2*zin;
			q2[g] = c2;
		}
	}


	// cycle over energy groups
	#ifdef INTEL
	#pragma vector
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		// load total cross section
		sigT[g] = S[QSR_id].sigT[g];

		// calculate common values for efficiency
		tau[g] = sigT[g] * ds;
		sigT2[g] = sigT[g] * sigT[g];
	}

	// cycle over energy groups
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		#ifdef TABLE
		expVal[g] = interpolateTable( table, tau[g] );  
		#else
		expVal[g] = 1.f - expf( -tau[g] ); // exp is faster on many architectures
		#endif
	}

	// Flux Integral

	// Re-used Term
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		reuse[g] = tau[g] * (tau[g] - 2.f) + 2.f * expVal[g] 
			/ (sigT[g] * sigT2[g]); 
	}

	//#pragma vector alignednontemporal
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		// add contribution to new source flux
		flux_integral[g] = (q0[g] * tau[g] + (sigT[g] * state_flux[g] - q0[g])
				* expVal[g]) / sigT2[g] + q1[g] * mu * reuse[g] + q2[g] * mu2 
			* (tau[g] * (tau[g] * (tau[g] - 3.f) + 6.f) - 6.f * expVal[g]) 
			/ (3.f * sigT2[g] * sigT2[g]);
	}

	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		// Prepare tally
		tally[g] = weight * flux_integral[g];
	}

	#ifdef OPENMP
	omp_set_lock(S[QSR_id].locks + FAI_id);
	#endif

	#ifdef INTEL
	#pragma vector
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		FSR_flux[g] += tally[g];
	}

	#ifdef OPENMP
	omp_unset_lock(S[QSR_id].locks + FAI_id);
	#endif

	// Term 1
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		t1[g] = q0[g] * expVal[g] / sigT[g];  
	}
	// Term 2
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		t2[g] = q1[g] * mu * (tau[g] - expVal[g]) / sigT2[g]; 
	}
	// Term 3
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		t3[g] =	q2[g] * mu2 * reuse[g];
	}
	// Term 4
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		t4[g] = state_flux[g] * (1.f - expVal[g]);
	}
	// Total psi
	#ifdef INTEL
	#pragma vector aligned
	#elif defined IBM
	#pragma vector_level(10)
	#endif
	for( int g = 0; g < egroups; g++)
	{
		state_flux[g] = t1[g] + t2[g] + t3[g] + t4[g];
	}
}	

/* Interpolates a formed exponential table to compute ( 1- exp(-x) )
 *  at the desired x value */
float interpolateTable( Table * restrict table, float x)
{
	// check to ensure value is in domain
	if( x > table->maxVal )
		return 1.0f;
	else
	{
		int interval = (int) ( x / table->dx + 0.5f * table->dx );
		/*
		   if( interval >= table->N || interval < 0)
		   {
		   printf( "Interval = %d\n", interval);
		   printf( "N = %d\n", table->N);
		   printf( "x = %f\n", x);
		   printf( "dx = %f\n", table->dx);
		   exit(1);
		   }
		   */
		interval = interval * 2;
		float slope = table->values[ interval ];
		float intercept = table->values[ interval + 1 ];
		float val = slope * x + intercept;
		return val;
	}
}

void print_state_flux( float * flux, int egroups )
{
	for( int i = 0; i < egroups; i++ )
	{
		printf("%.5e ", flux[i]);
	}
	printf("\n");
}
