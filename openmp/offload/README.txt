
	       README for Intel(R) Offload Runtime Library
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


Software Requirements
=====================

Intel(R) Offload Runtime Library requires additional software:

1) Intel(R) OpenMP* Runtime Library.  You can either download the source
code for that (from openmprtl.org or openmp.llvm.org) or simply use the
compiled version distributed with the Intel compilers.
2) Intel(R) COI Runtime Library and Intel(R) MYO Runtime Library.  These
libraries are part of Intel(R) Manycore Platform Software Stack (MPSS).  You
can download MPSS source code or binaries from
software.intel.com/en-us/articles/intel-manycore-platform-software-stack-mpss.
Binaries include host libraries for Intel(R) 64 Architecture and target
libraries for Intel(R) Many Integrated Core Architecture.

Also you will require all of the libraries that enable the target code to run
on device.  If you target the Intel(R) Xeon Phi (TM) coprocessor, these
libraries can be taken from MPSS too.


How to Build the Intel(R) Offload Runtime Library
=================================================

The Makefile at the top-level will attempt to detect what it needs to
build the Intel(R) Offload Runtime Library.  To see the default settings,
type:

make info

You can change the Makefile's behavior with the following options:

root_dir:	      The path to the top-level directory containing the
		      top-level Makefile.  By default, this will take on the
		      value of the current working directory.

build_dir:	      The path to the build directory.  By default, this will
		      take on value [root_dir]/build.

mpss_dir:	      The path to the Intel(R) Manycore Platform Software
		      Stack install directory.  By default, this will take on
		      the value of operating system's root directory.

libiomp_host_dir:     The path to the host Intel(R) OpenMP* Runtime Library.
		      This option is required when the host compiler is other
		      than icc.

libiomp_target_dir:   The path to the target Intel(R) OpenMP* Runtime
		      Library.  This option is required when the target
		      compiler is other than icc.

omp_header_dir:       The path to the header file <omp.h> of Intel(R) OpenMP*
		      Runtime Library.  This option is required if either host
		      or target compiler is other than icc.

os_host:	      Operating system on host.  Currently supports only
		      "linux" which is set by default.

os_target:	      Operating system on target device.  Currently supports
		      only "linux" which is set by default.

compiler_host:	      Which compiler to use for the build of the host part.
		      Defaults to "gcc"*.  Also supports "icc" and "clang"*.
		      You should provide the full path to the compiler or it
		      should be in the user's path.

compiler_host:	      Which compiler to use for the build of the target part.
		      Defaults to "gcc"*.  Also supports "icc" and "clang"*.
		      You should provide the full path to the compiler or it
		      should be in the user's path.

options_host:	      Additional options for the host compiler.

options_target:       Additional options for the target compiler.

To use any of the options above, simple add <option_name>=<value>.  For
example, if you want to build with icc instead of gcc, type:

make compiler_host=icc compiler_target=icc


Supported RTL Build Configurations
==================================

Supported Architectures: Intel(R) 64, and Intel(R) Many Integrated
Core Architecture

	      ---------------------------------------------
	      |   icc/icl     |    gcc      |    clang    |
--------------|---------------|---------------------------|
| Linux* OS   |      Yes      |     Yes(1)  |     Yes(1)  |
| OS X*       |       No      |      No     |      No     |
| Windows* OS |       No      |      No     |      No     |
-----------------------------------------------------------

(1) Liboffload requires _rdtsc intrinsic, which may be unsupported by some
    versions of compiler.  In this case you need to include src/rdtsc.h
    manually by using Makefile options options_host and options_target:

    make options_host="-include src/rdtsc.h" options_target="-include src/rdtsc.h"

-----------------------------------------------------------------------

Notices
=======

*Other names and brands may be claimed as the property of others.
