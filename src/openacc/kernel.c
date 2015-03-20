#include "SimpleMOC-kernel_header.h"

// removed:
//     SIMD_Vectors simd_vecs = allocate_simd_vectors(egroups);
//     float * flux_state = (float *) malloc(
//         egroups * sizeof(float));



void run_kernel( 
    int   _source_regions,
    int   _fine_axial_intervals,
    long  _segments,
    int   _egroups,
    int   _nthreads,
    int   _n_flux_states,
    float (* restrict fine_flux)[_fine_axial_intervals][_egroups], 
    float (* restrict fine_source)[_fine_axial_intervals][_egroups],
    float (* restrict sigT_arr)[_egroups],
    float (* restrict flux_states)[_egroups],
    int   (* restrict randIdx)[3])
{
  const int source_regions = _source_regions;
  const int fine_axial_intervals = _fine_axial_intervals;
  const long segments = _segments;
  const int egroups = _egroups;
  const int nthreads = _nthreads;
  const int n_flux_states = _n_flux_states;

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

  // Enter Parallel Region
#pragma acc data \
  copyin(\
      randIdx[0:segments][0:2], \
      fine_flux[0:source_regions][0:fine_axial_intervals][0:egroups], \
      fine_source[0:source_regions][0:fine_axial_intervals][0:egroups], \
      sigT_arr[0:source_regions][0:egroups], \
      ), \
  copy( \
      flux_states[0:n_flux_states][0:egroups], \
      )
  {

    // Enter OMP For Loop over Segments
#pragma acc kernels for
    for( long i = 0; i < segments; i++ )
    {
      // Pick random state flux vector
      const int FS_id = randIdx[i][0] % n_flux_states;

      // Pick Random QSR
      const int SR_id = randIdx[i][1] % source_regions;

      // Pick Random Fine Axial Interval
      const int FAI_id = randIdx[i][2] % fine_axial_intervals;

#pragma acc for
      for (int g=0; g < egroups; g++) 
      {

        // Attenuate Segment
        //attenuate_segment( I, S, QSR_id, FAI_id, flux_state,
        //    &simd_vecs, table);
        // Unload local vector vectors
        float q0;
        float q1;
        float q2;
        float sigT;
        float tau;
        float sigT2;
        float expVal;
        float reuse;
        float flux_integral;
        float tally;
        float t1;
        float t2;
        float t3;
        float t4;

        // Pointer to the state flux
        float * flux_state = flux_states[FS_id] + g;

        // Pointer to fine source region flux
        float * FSR_flux = fine_flux[SR_id][FAI_id] + g;

        if( FAI_id == 0 )
        {
          // load neighboring sources
          const float y2 = fine_source[SR_id][FAI_id  ][g];
          const float y3 = fine_source[SR_id][FAI_id+1][g];

          // do linear "fitting"
          const float c0 = y2;
          const float c1 = (y3 - y2) / dz;

          // calculate q0, q1, q2
          q0 = c0 + c1*zin;
          q1 = c1;
          q2 = 0;
        }
        else if ( FAI_id == fine_axial_intervals - 1 )
        {
          // load neighboring sources
          const float y1 = fine_source[SR_id][FAI_id-1][g];
          const float y2 = fine_source[SR_id][FAI_id  ][g];

          // do linear "fitting"
          const float c0 = y2;
          const float c1 = (y2 - y1) / dz;

          // calculate q0, q1, q2
          q0 = c0 + c1*zin;
          q1 = c1;
          q2 = 0;
        }
        else
        {
          // load neighboring sources
          const float y1 = fine_source[SR_id][FAI_id-1][g];
          const float y2 = fine_source[SR_id][FAI_id  ][g];
          const float y3 = fine_source[SR_id][FAI_id+1][g];

          // do quadratic "fitting"
          const float c0 = y2;
          const float c1 = (y1 - y3) / (2.f*dz);
          const float c2 = (y1 - 2.f*y2 + y3) / (2.f*dz*dz);

          // calculate q0, q1, q2
          q0 = c0 + c1*zin + c2*zin*zin;
          q1 = c1 + 2.f*c2*zin;
          q2 = c2;
        }

        // load total cross section
        sigT = sigT_arr[SR_id][g];

        // calculate common values for efficiency
        tau = sigT * ds;
        sigT2 = sigT * sigT;

        //expVal = interpolateTable( table, tau );  
        expVal = 1.f - exp( -tau ); // exp is faster on many architectures

        // Re-used Term
        reuse = tau * (tau - 2.f) + 2.f * expVal / (sigT * sigT2); 
        //
        // add contribution to new source flux
        flux_integral = (q0 * tau + (sigT * *flux_state - q0) * expVal) 
          / sigT2 + q1 * mu * reuse + q2 * mu2 
          * (tau * (tau * (tau - 3.f) + 6.f) - 6.f * expVal) 
          / (3.f * sigT2 * sigT2);
      }

      tally = weight * flux_integral;

      // WHAT TO DO HERE?
#ifdef OPENMP
      omp_set_lock(S[QSR_id].locks + FAI_id);
#endif
      for( int g = 0; g < egroups; g++)
      {
        FSR_flux[g] += tally[g];
      }
#ifdef OPENMP
      omp_unset_lock(S[QSR_id].locks + FAI_id);
#endif

      // Terms 1, 2, 3, and 4
      t1 = q0 * expVal / sigT;  
      t2 = q1 * mu * (tau - expVal) / sigT2; 
      t3 =	q2 * mu2 * reuse;
      t4 = flux_state * (1.f - expVal);

      *flux_state = t1 + t2 + t3 + t4;
    }
  }
}
