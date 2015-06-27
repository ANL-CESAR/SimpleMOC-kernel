#include "SimpleMOC-kernel_header.h"

/* My parallelization scheme here is to basically have a single
 * block be a geometrical segment, with each thread within the
 * block represent a single energy phase. On the CPU, the
 * inner SIMD-ized loop is over energy (i.e, 100 energy groups).
 * This should allow for each BLOCK to have:
 * 		- A single state variable for the RNG
 * 		- A set of __shared__ SIMD vectors, each thread id being its idx
 */

__global__ void run_kernel( Input I, Source *  S,
		Source_Arrays SA, Table *  table, curandState *  state,
		float *  state_fluxes, int N_state_fluxes,
		unsigned long long * vhash)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x; // geometric segment	

	if( blockId >= I.segments / I.seg_per_thread )
		return;

	// Assign RNG state
	curandState *  localState = &state[blockId % I.streams];

	// Assign multiple segments to each block rather than 1:1
	// This makes things significantly faster as blocks have more work
	blockId *= I.seg_per_thread;
	blockId--;

	int g = threadIdx.x; // Each energy group (g) is one thread in a block

	// Thread Local (i.e., specific to E group) variables
	// Similar to SIMD vectors in CPU code
	float q0           ;
	float q1           ;
	float q2           ;
	float sigT         ;
	float tau          ;
	float sigT2        ;
	float expVal       ;
	float reuse        ;
	float flux_integral;
	float tally        ;
	float t1           ;
	float t2           ;
	float t3           ;
	float t4           ;

	// Randomized variables (common accross all thread within block)
	extern __shared__ int shm[];
	int *  state_flux_id = &shm[0];
	int *  QSR_id = &shm[I.seg_per_thread];
	int *  FAI_id = &shm[I.seg_per_thread * 2];

	if( threadIdx.x == 0 ) // Specifies only done once per CUDA block
	{
		for( int i = 0; i < I.seg_per_thread; i++ ) // loops through segments for block
		{
			#ifdef VERIFY
			// Sets randomized ID's deterministically based on segment ID
			state_flux_id[i] = curand(localState) % N_state_fluxes;
			QSR_id[i] = ( blockId + 1 + i ) % I.source_3D_regions;
			FAI_id[i] = ( blockId + 1 + i ) % I.fine_axial_intervals;
			#else
			state_flux_id[i] = curand(localState) % N_state_fluxes;
			QSR_id[i] = curand(localState) % I.source_3D_regions;
			FAI_id[i] = curand(localState) % I.fine_axial_intervals;
			#endif
		}
	}

	__syncthreads();

	#ifdef VERIFY
	unsigned long long thread_local_hash = 0;
	#endif

	for( int i = 0; i < I.seg_per_thread; i++ )
	{
		blockId++;

		float *  state_flux = &state_fluxes[state_flux_id[i]];

		#ifdef VERIFY
		state_flux[g] = 1.0;
		#endif

		__syncthreads();

		//////////////////////////////////////////////////////////
		// Attenuate Segment
		//////////////////////////////////////////////////////////

		// Some placeholder constants - In the full app some of these are
		// calculated based off position in geometry. This treatment
		// shaves off a few FLOPS, but is not significant compared to the
		// rest of the function.
		float dz = 0.1f;
		float zin = 0.3f; 
		float weight = 0.5f;
		float mu = 0.9f;
		float mu2 = 0.3f;
		float ds = 0.7f;

		const int egroups = I.egroups;

		// load fine source region flux vector
		float *  FSR_flux = &SA.fine_flux_arr[ S[QSR_id[i]].fine_flux_id + FAI_id[i] * egroups];

		if( FAI_id[i] == 0 )
		{
			float *   f2 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i])*egroups];
			float *   f3 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i]+1)*egroups];
			// cycle over energy groups
			// load neighboring sources
			float y2 = __ldg(&f2[g]);
			float y3 = __ldg(&f3[g]);

			// do linear "fitting"
			float c0 = y2;
			float c1 = (y3 - y2) / dz;

			// calculate q0, q1, q2
			q0 = c0 + c1*zin;
			q1 = c1;
			q2 = 0;
		}
		else if ( FAI_id[i] == I.fine_axial_intervals - 1 )
		{
			float *   f1 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i]-1)*egroups];
			float *   f2 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i])*egroups];
			// cycle over energy groups
			// load neighboring sources
			float y1 = __ldg(&f1[g]);
			float y2 = __ldg(&f2[g]);

			// do linear "fitting"
			float c0 = y2;
			float c1 = (y2 - y1) / dz;

			// calculate q0, q1, q2
			q0 = c0 + c1*zin;
			q1 = c1;
			q2 = 0;
		}
		else
		{
			float *   f1 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i]-1)*egroups];
			float *   f2 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i])*egroups];
			float *   f3 = &SA.fine_source_arr[ S[QSR_id[i]].fine_source_id + (FAI_id[i]+1)*egroups];
			// cycle over energy groups
			// load neighboring sources
			float y1 = __ldg(&f1[g]); 
			float y2 = __ldg(&f2[g]);
			float y3 = __ldg(&f3[g]);

			// do quadratic "fitting"
			float c0 = y2;
			float c1 = (y1 - y3) / (2.f*dz);
			float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

			// calculate q0, q1, q2
			q0 = c0 + c1*zin + c2*zin*zin;
			q1 = c1 + 2.f*c2*zin;
			q2 = c2;
		}

		// load total cross section
		sigT = __ldg(&SA.sigT_arr[ S[QSR_id[i]].sigT_id + g]);

		// calculate common values for efficiency
		tau = sigT * ds;
		sigT2 = sigT * sigT;

		#ifdef TABLE
		interpolateTable( table, tau, &expVal );  
		#else
		expVal = 1.f - expf( -tau); // EXP function is fater than table lookup
		#endif

		// Flux Integral

		// Re-used Term
		reuse = tau * (tau - 2.f) + 2.f * expVal
			/ (sigT * sigT2); 

		// add contribution to new source flux
		flux_integral = (q0 * tau + (sigT * __ldg(&state_flux[g]) - q0)
				* expVal) / sigT2 + q1 * mu * reuse + q2 * mu2 
			* (tau * (tau * (tau - 3.f) + 6.f) - 6.f * expVal) 
			/ (3.f * sigT2 * sigT2);

		// Prepare tally
		tally = weight * flux_integral;

		// SHOULD BE ATOMIC HERE!
		//FSR_flux[g] += tally;
		atomicAdd(&FSR_flux[g], (float) tally);

		// Term 1
		t1 = q0 * expVal / sigT;  
		// Term 2
		t2 = q1 * mu * (tau - expVal) / sigT2; 
		// Term 3
		t3 =	q2 * mu2 * reuse;
		// Term 4
		t4 = state_flux[g] * (1.f - expVal);
		// Total psi
		state_flux[g] = t1 + t2 + t3 + t4;

		#ifdef VERIFY
		thread_local_hash += hash(state_flux[g]);
		#endif
	}
	
	#ifdef VERIFY
	__syncthreads();
	__shared__ unsigned long long block_hash;
	if( threadIdx.x == 0 )
		block_hash = 0;
	for( int i = 0; i < I.egroups; i++ )
	{
		if( threadIdx.x == i )
		{
			block_hash += thread_local_hash;
		}
		__syncthreads();
	}
	__syncthreads();
	//atomicAdd(&block_hash, (unsigned long long ) thread_local_hash);
	if( threadIdx.x == 0 ) // Specifies only done once per CUDA block
	{
		atomicAdd(vhash, block_hash);
	}
	#endif
}	

/* Interpolates a formed exponential table to compute ( 1- exp(-x) )
 *  at the desired x value */
__device__ void interpolateTable(Table *  table, float x, float *  out)
{
	// check to ensure value is in domain
	if( x > table->maxVal )
		*out = 1.0f;
	else
	{
		int interval = (int) ( x / table->dx + 0.5f * table->dx );
		interval = interval * 2;
		float slope = table->values[ interval ];
		float intercept = table->values[ interval + 1 ];
		float val = slope * x + intercept;
		*out = val;
	}
}

__device__ unsigned int hash( float f )
{
	float rounded_up = ceilf(f * 10000.) / 10000.; 
	unsigned int ui;
	memcpy( &ui, &rounded_up, sizeof( float ) );
	//return ui & 0xfffff000;
	//return ui;
	return 1;
}
