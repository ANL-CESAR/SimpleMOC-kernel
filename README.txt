===============================================================================
    
              _____ _                 _      __  __  ____   _____ 
             / ____(_)               | |    |  \/  |/ __ \ / ____|
            | (___  _ _ __ ___  _ __ | | ___| \  / | |  | | |     
             \___ \| | '_ ` _ \| '_ \| |/ _ \ |\/| | |  | | |     
             ____) | | | | | | | |_) | |  __/ |  | | |__| | |____ 
            |_____/|_|_| |_| |_| .__/|_|\___|_|  |_|\____/ \_____|
                               | |                                
                               |_|                                
                           _  __                    _ 
                          | |/ /___ _ __ _ __   ___| |
                          | ' // _ \ '__| '_ \ / _ \ |
                          | . \  __/ |  | | | |  __/ |
                          |_|\_\___|_|  |_| |_|\___|_|

==============================================================================
Contact Information
==============================================================================

Organizations:     Computational Reactor Physics Group
                   Massachusetts Institute of Technology

                   Center for Exascale Simulation of Advanced Reactors (CESAR)
                   Argonne National Laboratory

Development Leads: John Tramm     <jtramm@mit.edu>
                   Geoffrey Gunow <geogunow@mit.edu>
    
===============================================================================
What is SimpleMOC-kernel?
===============================================================================

SimpleMOC-kernel represents the core computational of a larger application
(SimpleMOC). This app was written in order to abstract away much of the
complexity of the full application in order to facilitate easier porting of
the code and enable more transparent analysis techniques on high performance
architectures.

The scope of this kernel is essentially the inner-loop of SimpleMOC, i.e., the
attentuation of neutron fluxes across an individual geometrical segment.
This kernel captures approximately 90% of the runtime of the full application,
and is therefore useful for analyzing optimization methods and performance
implications for exascale supercomputer architectures.

==============================================================================
Quick Start Guide
==============================================================================

Download----------------------------------------------------------------------

	For the most up-to-date version of SimpleMOC-kernel, we recommend that you
	download from our git repository. This can be accomplished via
	cloning the repository from the command line, or by downloading a zip
	from our github page.

	Git Repository Clone:
		
		Use the following command to clone SimpleMOC-kernel to your machine:

		>$ git clone https://github.com/ANL-CESAR/SimpleMOC-kernel.git

		Once cloned, you can update the code to the newest version
		using the following command (when in the SimpleMOC-kernel directory):

		>$ git pull

Compilation-------------------------------------------------------------------

	To compile SimpleMOC-kernel with default (serial mode) settings,
	use the following command:

	>$ make

	To disable shared memory (OpenMP) paralleism, set the OpenMP flag to
	"no" in the makefile before building.  See below for more details
	regarding advanced compilation options.

Running SimpleMOC-kernel-------------------------------------------------------

	To run SimpleMOC-kernel with default settings, use the following command:

	>$ ./SimpleMOC-kernel

	For non-default settings, SimpleMOC-kernel supports the following
	command line options:

	Usage: ./SimpleMOC-kernel <options>
	Options include:
	  -t <threads>        Number of OpenMP threads to run
	  -s <segments>       Number of segments to process
	  -e <energy groups>  Number of energy groups
	  -p <PAPI event>     PAPI event name to count (1 only)

	If not options are specified, then a default set of parameters will
	automatically be run. These parameters reflect the approximate per node
	work load for a full core reactor simulation (the the number of geometry
	segments has been signficantly reduced to reduce runtime while preserving
	the computational profile).

==============================================================================
Advanced Compilation, Debugging, Optimization, and Profiling
==============================================================================

There are a number of switches that can be set at the top of the makefile
to enable MPI and OpenMP parallelism, along with more advanced compilation
features.

Here is a sample of the control panel at the top of the makefile:

COMPILER    = gnu
OPENMP      = no
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
PAPI        = no

Explanation of Flags:

COMPILER <gnu, intel, ibm> - This selects your compiler.

OpenMP - Enables OpenMP support in the code. By default, the code will
         run using the maximum number of threads on the system, unless
         otherwise specified with the "-t" command line argument.

OPTIMIZE - Adds compiler optimization flag "-O3".

DEBUG - Adds the compiler flag "-g".

PROFILE - Adds the compiler flag "-pg".

PAPI - Enables PAPI support in the code. You may need to alter the makefile
       or your environment to ensure proper linking with the PAPI library.
       See PAPI section below for more details.

===============================================================================
SimpleMOC-kernel Strawman Reactor Defintion
===============================================================================

For the purposes of simplicity this mini-app uses a conservative "strawman"
reactor model to represent a good target problem for full core reactor
simualations to be run on exascale class supercomputers. Arbitrary
user-defined geometries are not supported.

===============================================================================
