
               README for Intel(R) OpenMP* Runtime Library
               ===========================================

How to Build Documentation
==========================

The main documentation is in Doxygen* format, and this distribution
should come with pre-built PDF documentation in doc/Reference.pdf.  
However, an HTML version can be built by executing:

% doxygen doc/doxygen/config 

in this directory.

That will produce HTML documentation in the doc/doxygen/generated
directory, which can be accessed by pointing a web browser at the
index.html file there.

If you don't have Doxygen installed, you can download it from
www.doxygen.org.


How to Build the Intel(R) OpenMP* Runtime Library
=================================================

The Makefile at the top-level will attempt to detect what it needs to
build the Intel(R) OpenMP* Runtime Library.  To see the default settings, 
type:

make info

You can change the Makefile's behavior with the following options:

omp_root:    The path to the top-level directory containing the top-level
	     Makefile.  By default, this will take on the value of the 
	     current working directory.

omp_os:      Operating system.  By default, the build will attempt to 
	     detect this. Currently supports "linux", "freebsd", "macos", and 
	     "windows".

arch:        Architecture. By default, the build will attempt to 
	     detect this if not specified by the user. Currently 
	     supported values are
                 "32" for IA-32 architecture 
                 "32e" for Intel(R) 64 architecture
                 "mic" for Intel(R) Many Integrated Core Architecture

             If "mic" is specified then "icc" will be used as the
	     compiler, and appropriate k1om binutils will be used. The
	     necessary packages must be installed on the build machine
	     for this to be possible (but an Intel(R) Xeon Phi(TM)
	     coprocessor card is not required to build the library).

compiler:    Which compiler to use for the build.  Defaults to "icc" 
	     or "icl" depending on the value of omp_os. Also supports 
	     some versions of "gcc"* when omp_os is "linux". The selected 
	     compiler should be installed and in the user's path. The 
	     corresponding Fortran compiler should also be in the path. 
	     See "Supported RTL Build Configurations" below for more 
	     information on compiler versions.

mode:        Library mode: default is "release".  Also supports "debug".

To use any of the options above, simple add <option_name>=<value>.  For 
example, if you want to build with gcc instead of icc, type:

make compiler=gcc

There is also an experimental CMake build system. This is *not* yet
supported for production use and resulting binaries have not been checked
for compatibility.

On OS X* machines, it is possible to build universal (or fat) libraries which
include both IA-32 architecture and Intel(R) 64 architecture objects in a
single archive; just build the 32 and 32e libraries separately, then invoke 
make again with a special argument as follows:

make compiler=clang build_args=fat

Supported RTL Build Configurations
==================================

Supported Architectures: IA-32 architecture, Intel(R) 64, and 
Intel(R) Many Integrated Core Architecture

              ----------------------------------------------
              |   icc/icl     |    gcc      |   clang      |
--------------|---------------|----------------------------|
| Linux* OS   |   Yes(1,5)    |  Yes(2,4)   | Yes(4,6,7)   |
| FreeBSD*    |   No          |  No         | Yes(4,6,7,8) |
| OS X*       |   Yes(1,3,4)  |  No         | Yes(4,6,7)   |
| Windows* OS |   Yes(1,4)    |  No         | No           |
------------------------------------------------------------

(1) On IA-32 architecture and Intel(R) 64, icc/icl versions 12.x are 
    supported (12.1 is recommended).
(2) GCC* version 4.6.2 is supported.
(3) For icc on OS X*, OS X* version 10.5.8 is supported.
(4) Intel(R) Many Integrated Core Architecture not supported.
(5) On Intel(R) Many Integrated Core Architecture, icc/icl versions 13.0 
    or later are required.
(6) Clang* version 3.3 is supported.
(7) Clang* currently does not offer a software-implemented 128 bit extended 
    precision type.  Thus, all entry points reliant on this type are removed
    from the library and cannot be called in the user program.  The following
    functions are not available:
    __kmpc_atomic_cmplx16_*
    __kmpc_atomic_float16_*
    __kmpc_atomic_*_fp
(8) Community contribution provided AS IS, not tested by Intel.

Front-end Compilers that work with this RTL
===========================================

The following compilers are known to do compatible code generation for
this RTL:  icc/icl, gcc.  See the documentation for more detail.

-----------------------------------------------------------------------

Notices
=======

*Other names and brands may be claimed as the property of others.
