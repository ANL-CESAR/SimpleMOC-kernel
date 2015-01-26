#include "SimpleMOC-kernel_header.h"

int main( int argc, char * argv[] )
{
  int version = 1;

  srand(time(NULL));

  Input I = set_default_input();
  read_CLI( argc, argv, &I );

  logo(version);

  print_input_summary(I);

  center_print("INITIALIZATION", 79);
  border_print();

  // Setup OCCA device and info
  int outer_dim = sqrt(I.segments);
  int inner_dim = I.egroups;

  const char *device_infos = "mode = OpenMP";
  occaDevice device = occaGetDevice(device_infos);

  occaKernelInfo kinfo = occaGenKernelInfo();
  occaKernelInfoAddDefine(kinfo, "outerDim0", occaLong(outer_dim));
  occaKernelInfoAddDefine(kinfo, "outerDim1", occaLong(outer_dim));
  occaKernelInfoAddDefine(kinfo, "innerDim0", occaLong(inner_dim));

  // Build OCCA kernel
  occaKernel run_kernel = occaBuildKernelFromSource(device,
                                                    "run_kernel.okl",
                                                    "run_kernel",
                                                    kinfo);
  printf("Compiled run_kernel.okl\n");

  // Build Source Data
  printf("Building Source Data Arrays...\n");
  Source_Arrays SA_h;
  OCCA_Source_Arrays SA_d;
  Source * sources_h = initialize_sources(I, &SA_h);
  occaMemory sources_d = initialize_occa_sources( I, &SA_h, &SA_d, sources_h,
                                                  device);
  occaDeviceFinish(device);

  // Setup RNG on Device
  // NOTE - CUDA RNG is not going to work with OCCA - so we're going to revert
  // to the standard linear congruential generation algorithm while preserving
  // the limited state streams concept (i.e., # RNG states << # GPU threads)
  printf("Setting up RNG...\n");
  unsigned long * RNG_states_h = (unsigned long *) malloc( I.streams *
                                                           sizeof(unsigned long));
  // Init states to something
  unsigned long time_of_exec = time(NULL);
  for( int i = 0; i < I.streams; i++ )
    RNG_states_h[i] = time_of_exec + i + 1;
  occaMemory RNG_states = occaDeviceMalloc(device, I.streams * sizeof(unsigned
                                                                      long), RNG_states_h);
  free(RNG_states_h); // as we don't need host states anymore
  occaDeviceFinish(device);

  // Allocate Some Flux State vectors to randomly pick from
  printf("Setting up Flux State Vectors...\n");

  int N_flux_states = 10000;
  assert( I.segments >= N_flux_states );
  float * flux_state_h = (float *) malloc( N_flux_states * I.egroups * sizeof(float));
  for( int i = 0; i < N_flux_states * I.egroups; i++ )
  {
	  flux_state_h[i] = (float) rand() / RAND_MAX;
  }
  occaMemory flux_states = occaDeviceMalloc(device,
                                            N_flux_states * I.egroups * sizeof(float),
                                            flux_state_h);
  free(flux_state_h);

  printf("Initialization Complete.\n");
  border_print();
  center_print("SIMULATION", 79);
  border_print();
  occaDeviceFinish(device);
  printf("Attentuating fluxes across segments...\n");

  // timer variables
  struct timeval start, end;

  // Run Simulation Kernel Loop
  gettimeofday(&start, NULL);

  occaKernelRun(run_kernel,
                occaStruct(&I, sizeof(I)),
                sources_d,
                SA_d.fine_flux_arr,
                SA_d.fine_source_arr,
                SA_d.sigT_arr,
                RNG_states,
                flux_states,
                occaInt(N_flux_states));

  gettimeofday(&end, NULL);
  double time = (end.tv_sec - start.tv_sec) + (end.tv_usec -
                                               start.tv_usec)/1000000.;
  occaDeviceFinish(device);

  // Copy final data back to host, for kicks.
  float * host_flux_states = (float*) malloc(N_flux_states * I.egroups *
                                             sizeof(float));
  occaCopyMemToPtr(host_flux_states, flux_states, N_flux_states * I.egroups *
                   sizeof(float), 0);

  printf("Simulation Complete.\n");

  border_print();
  center_print("RESULTS SUMMARY", 79);
  border_print();

  double tpi = (time / (double)I.segments / (double) I.egroups) * 1.0e9;
  printf("%-25s%.3f seconds\n", "Runtime:", time );
  printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
  border_print();

  return 0;
}
