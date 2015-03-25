#include "SimpleMOC-kernel_header.h"

// removed:
//     SIMD_Vectors simd_vecs = allocate_simd_vectors(egroups);
//     float * state_flux = (float *) malloc(
//         egroups * sizeof(float));



void run_kernel( 
    int      source_regions,
    int      fine_axial_intervals,
    long     segments,
    int      egroups,
    int      nthreads,
    int      n_state_fluxes,
    float    (* restrict fine_flux_arr)[fine_axial_intervals][egroups], 
    float    (* restrict fine_source_arr)[fine_axial_intervals][egroups],
    float    (* restrict sigT_arr)[egroups],
    float    (* restrict state_flux_arr)[egroups],
    unsigned (* restrict randIdx)[3]
    )
{
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
      source_regions, \
      fine_axial_intervals, \
      segments, \
      egroups, \
      nthreads, \
      n_state_fluxes, \
      dz, \
      zin, \
      weight, \
      mu, \
      mu2, \
      ds, \
      randIdx[0:segments][0:2], \
      fine_source_arr[0:source_regions][0:fine_axial_intervals][0:egroups], \
      sigT_arr[0:source_regions][0:egroups] \
      ) \
  copy( \
      fine_flux_arr[0:source_regions][0:fine_axial_intervals][0:egroups], \
      state_flux_arr[0:n_state_fluxes][0:egroups] \
      )
  {

    // Enter OMP For Loop over Segments
    const long _segments = segments;
#pragma acc kernels for
    for( long i = 0; i < _segments; i++ )
    {
      // Pick random state flux vector
      const int SF_id = randIdx[i][0] % n_state_fluxes;

      // Pick Random QSR
      const int QSR_id = randIdx[i][1] % source_regions;

      // Pick Random Fine Axial Interval
      const int FAI_id = randIdx[i][2] % fine_axial_intervals;

      const int _egroups = egroups;
#pragma acc for
      for (int g=0; g < _egroups; g++) 
      {
        // Attenuate Segment
        //attenuate_segment( I, S, QSR_id, FAI_id, state_flux,
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
        float * state_flux = &state_flux_arr[SF_id][g];

        // Pointer to fine source region flux
        float * FSR_flux = &fine_flux_arr[QSR_id][FAI_id][g];

        if( FAI_id == 0 )
        {
          // load neighboring sources
          const float y2 = fine_source_arr[QSR_id][FAI_id  ][g];
          const float y3 = fine_source_arr[QSR_id][FAI_id+1][g];

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
          const float y1 = fine_source_arr[QSR_id][FAI_id-1][g];
          const float y2 = fine_source_arr[QSR_id][FAI_id  ][g];

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
          const float y1 = fine_source_arr[QSR_id][FAI_id-1][g];
          const float y2 = fine_source_arr[QSR_id][FAI_id  ][g];
          const float y3 = fine_source_arr[QSR_id][FAI_id+1][g];

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
        sigT = sigT_arr[QSR_id][g];

        // calculate common values for efficiency
        tau = sigT * ds;
        sigT2 = sigT * sigT;

        //expVal = interpolateTable( table, tau );  
        expVal = 1.f - exp( -tau ); // exp is faster on many architectures

        // Re-used Term
        reuse = tau * (tau - 2.f) + 2.f * expVal / (sigT * sigT2); 
        //
        // add contribution to new source flux
        flux_integral = (q0 * tau + (sigT * *state_flux - q0) * expVal) 
          / sigT2 + q1 * mu * reuse + q2 * mu2 
          * (tau * (tau * (tau - 3.f) + 6.f) - 6.f * expVal) 
          / (3.f * sigT2 * sigT2);

        tally = weight * flux_integral;

#pragma acc atomic update
        *FSR_flux += tally;

        // Terms 1, 2, 3, and 4
        t1 = q0 * expVal / sigT;  
        t2 = q1 * mu * (tau - expVal) / sigT2; 
        t3 =	q2 * mu2 * reuse;
        t4 = *state_flux * (1.f - expVal);

        //*state_flux = t1 + t2 + t3 + t4;

      } // END: for (int g=0; g < egroups; g++) 
    } // END: for( long i = 0; i < segments; i++ )
  } // END: #pragma acc data ...
} // END: void run_kernel( ... )
