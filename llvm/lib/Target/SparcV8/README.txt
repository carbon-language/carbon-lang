
SparcV8 backend skeleton
------------------------

This directory houses a 32-bit SPARC V8 backend employing an expander-based
instruction selector.  It is not yet functionally complete.  Watch
this space for more news coming soon!

Current expected test failures
------------------------------

The SparcV8 backend works on many simple C++ SingleSource codes. Here
are the known SingleSource failures:

	Regression/C++/EH/exception_spec_test
	Regression/C++/EH/throw_rethrow_test
	Benchmarks/Shootout-C++/moments
	Benchmarks/Shootout-C++/random

Here are the known MultiSource test failures, neglecting FreeBench,
MallocBench, and Prolangs-C++:

  Applications/lambda
  Applications/d/make_dparser
  Applications/hbd
  Applications/hexxagon
  Benchmarks/Fhourstones
  Benchmarks/McCat/03-testtrie
  Benchmarks/McCat/18-imp
  Benchmarks/Olden/tsp
  Benchmarks/Ptrdist/anagram
  Benchmarks/Prolangs-C/bison/mybison
  Benchmarks/Prolangs-C/fixoutput
  Benchmarks/Prolangs-C/gnugo
  Benchmarks/Prolangs-C/plot2fig

To-do
-----

* support shifts on longs
* support casting 64-bit integers to FP types
* support FP rem
* directly support select instructions

$Date$

