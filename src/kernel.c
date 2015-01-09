#include "SimpleMOC-kernel_header.h"

/* My parallelization scheme here is to basically have a single
 * block be a geometrical segment, with each thread within the
 * block represent a single energy phase. On the CPU, the
 * inner SIMD-ized loop is over energy (i.e, 100 energy groups).
 * This should allow for each BLOCK to have:
 * 		- A single state variable for the RNG
 * 		- A set of __shared__ SIMD vectors, each thread id being its idx
 */

__global__ void run_kernel( Input I, Source * S,
		Source_Arrays * SA, Table * table, curandState * state,
		float * state_fluxes, int N_state_fluxes)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x; // geometric segment	
	int threadId = blockId * blockDim.x + threadIdx.x;

	int g = threadIdx.x; // Each energy group (g) is one thread in a block

	// Assign shared SIMD vectors
	extern __shared__ float simd_shared_vecs[];
	float * s = simd_shared_vecs;
	float * q0 =            s; s+= I.egroups;
	float * q1 =            s; s+= I.egroups;
	float * q2 =            s; s+= I.egroups;
	float * sigT =          s; s+= I.egroups;
	float * tau =           s; s+= I.egroups;
	float * sigT2 =         s; s+= I.egroups;
	float * expVal =        s; s+= I.egroups;
	float * reuse =         s; s+= I.egroups;
	float * flux_integral = s; s+= I.egroups;
	float * tally =         s; s+= I.egroups;
	float * t1 =            s; s+= I.egroups;
	float * t2 =            s; s+= I.egroups;
	float * t3 =            s; s+= I.egroups;
	float * t4 =            s; s+= I.egroups;

	// Assign RNG state
	curandState * localState = &state[blockId];

	// Randomized variables (common accross all thread within block)
	__shared__ int state_flux_id;
	__shared__ int QSR_id;
	__shared__ int FAI_id;

	// Find State Flux Vector in global memory
	// (We are not concerned with coherency here as in actual
	// program threads would be organized in a more specific order)
	if( threadIdx.x == 0 )
		state_flux_id = curand(&localState) * N_state_fluxes;

	__syncthreads();
	float * state_flux = &state_fluxes[state_flux_id];

	// Pick Random QSR
	if( threadIdx.x == 0 )
		QSR_id = curand(&localState) * I.source_regions;

	// Pick Random Fine Axial Interval
	if( threadIdx.x == 0 )
		FAI_id = curand(&localState) * I.fine_axial_intervals;

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
	float * FSR_flux = &SA.fine_flux_arr[ S[QSR_id].fine_flux_id + FAI_id * egroups];

	if( FAI_id == 0 )
	{
		float * f2 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id)*egroups];
		float * f3 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id+1)*egroups];
		// cycle over energy groups
		// load neighboring sources
		float y2 = f2[g];
		float y3 = f3[g];

		// do linear "fitting"
		float c0 = y2;
		float c1 = (y3 - y2) / dz;

		// calculate q0, q1, q2
		q0[g] = c0 + c1*zin;
		q1[g] = c1;
		q2[g] = 0;
	}
	else if ( FAI_id == I.fine_axial_intervals - 1 )
	{
		float * f1 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id-1)*egroups];
		float * f2 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id)*egroups];
		// cycle over energy groups
		// load neighboring sources
		float y1 = f1[g];
		float y2 = f2[g];

		// do linear "fitting"
		float c0 = y2;
		float c1 = (y2 - y1) / dz;

		// calculate q0, q1, q2
		q0[g] = c0 + c1*zin;
		q1[g] = c1;
		q2[g] = 0;
	}
	else
	{
		float * f1 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id-1)*egroups];
		float * f2 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id)*egroups];
		float * f3 = &SA.fine_flux_arr[ S[QSR_id].fine_source_id + (FAI_id+1)*egroups];
		// cycle over energy groups
		// load neighboring sources
		float y1 = f1[g]; 
		float y2 = f2[g];
		float y3 = f3[g];

		// do quadratic "fitting"
		float c0 = y2;
		float c1 = (y1 - y3) / (2.f*dz);
		float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

		// calculate q0, q1, q2
		q0[g] = c0 + c1*zin + c2*zin*zin;
		q1[g] = c1 + 2.f*c2*zin;
		q2[g] = c2;
	}


	// load total cross section
	sigT[g] = SA.sigT_arr[ S[QSR_id].sigT_id + g];

	// calculate common values for efficiency
	tau[g] = sigT[g] * ds;
	sigT2[g] = sigT[g] * sigT[g];

	interpolateTable( table, tau[g], &expVal[g] );  

	// Flux Integral

	// Re-used Term
	reuse[g] = tau[g] * (tau[g] - 2.f) + 2.f * expVal[g] 
		/ (sigT[g] * sigT2[g]); 

	// add contribution to new source flux
	flux_integral[g] = (q0[g] * tau[g] + (sigT[g] * state_flux[g] - q0[g])
			* expVal[g]) / sigT2[g] + q1[g] * mu * reuse[g] + q2[g] * mu2 
		* (tau[g] * (tau[g] * (tau[g] - 3.f) + 6.f) - 6.f * expVal[g]) 
		/ (3.f * sigT2[g] * sigT2[g]);

	// Prepare tally
	tally[g] = weight * flux_integral[g];

	// SHOULD BE ATOMIC HERE!
	//FSR_flux[g] += tally[g];
	atomicAdd(&FSR_flux[g], (float) tally[g]);

	// Term 1
	t1[g] = q0[g] * expVal[g] / sigT[g];  
	// Term 2
	t2[g] = q1[g] * mu * (tau[g] - expVal[g]) / sigT2[g]; 
	// Term 3
	t3[g] =	q2[g] * mu2 * reuse[g];
	// Term 4
	t4[g] = state_flux[g] * (1.f - expVal[g]);
	// Total psi
	state_flux[g] = t1[g] + t2[g] + t3[g] + t4[g];
}	

/* Interpolates a formed exponential table to compute ( 1- exp(-x) )
 *  at the desired x value */
__device__ void interpolateTable(Table table, float x, float * out)
{
	// check to ensure value is in domain
	if( x > table.maxVal )
		*out = 1.0f;
	else
	{
		int interval = (int) ( x / table.dx + 0.5f * table.dx );
		interval = interval * 2;
		float slope = table.values[ interval ];
		float intercept = table.values[ interval + 1 ];
		float val = slope * x + intercept;
		*out = val;
	}
}
