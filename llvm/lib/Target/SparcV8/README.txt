
SparcV8 backend skeleton
------------------------

This directory houses a 32-bit SPARC V8 backend employing an expander-based
instruction selector.  It is not yet functionally complete.  Watch
this space for more news coming soon!

Current expected test failures
------------------------------

Here are the currently-expected SingleSource failures for V8
(Some C++ programs are crashing in libstdc++ at the moment;
I'm not sure why.)

  (llc) SingleSource/Regression/C++/EH/exception_spec_test
  (llc) SingleSource/Regression/C++/EH/throw_rethrow_test

Here are the currently-expected MultiSource failures for V8,
neglecting FreeBench, MallocBench, and Prolangs-C++:

  (llc,cbe) MultiSource/Applications/d/make_dparser
  (llc) MultiSource/Applications/hbd
  (llc,cbe) MultiSource/Applications/hexxagon
  (llc) MultiSource/Benchmarks/Fhourstones
  (llc,cbe) MultiSource/Benchmarks/McCat/03-testtrie
  (llc) MultiSource/Benchmarks/McCat/18-imp
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/bison/mybison
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/fixoutput
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/gnugo
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/plot2fig
  (llc,cbe) MultiSource/Benchmarks/Ptrdist/anagram

To-do
-----

* support shifts on longs
* support casting 64-bit integers to FP types
* support FP rem
* directly support select instructions

$Date$

