
Meta TODO list:
1. Create a new DAG -> DAG instruction selector, by adding patterns to the
   instructions.
2. ???
3. profit!


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

Here are the currently-expected MultiSource failures for V8:

  (llc,cbe) MultiSource/Applications/d/make_dparser
  (llc,cbe) MultiSource/Applications/hexxagon
  (llc) MultiSource/Benchmarks/Fhourstones
  (llc,cbe) MultiSource/Benchmarks/McCat/03-testtrie
  (llc) MultiSource/Benchmarks/McCat/18-imp
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/bison/mybison
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/fixoutput
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/gnugo
  (llc,cbe) MultiSource/Benchmarks/Prolangs-C/plot2fig
  (llc,cbe) MultiSource/Benchmarks/Ptrdist/anagram
  (llc,cbe) MultiSource/Benchmarks/FreeBench/analyzer
    * DANGER * analyzer will run the machine out of VM
  (I don't know whether the following fail in cbe:)
  (llc) MultiSource/Benchmarks/FreeBench/distray
  (llc) MultiSource/Benchmarks/FreeBench/fourinarow
  (llc) MultiSource/Benchmarks/FreeBench/pifft
  (llc) MultiSource/Benchmarks/MallocBench/gs
  (llc) MultiSource/Benchmarks/Prolangs-C++/deriv1
  (llc) MultiSource/Benchmarks/Prolangs-C++/deriv2

Known SPEC failures for V8 (probably not an exhaustive list):

  (llc) 134.perl
  (llc) 177.mesa
  (llc) 188.ammp -- FPMover bug?
  (llc) 256.bzip2
  (llc,cbe) 130.li
  (native,llc,cbe) 126.gcc
  (native,llc,cbe) 255.vortex

To-do
-----

* support shl on longs (fourinarow needs this)
* support casting 64-bit integers to FP types (fhourstones needs this)
* support FP rem (call fmod)

* Keep the address of the constant pool in a register instead of forming its
  address all of the time.

* Change code like this:
        or      %o0, %lo(.CPI_main_0), %o0
        ld      [%o0+0], %o0
  into:
        ld	[%o0+%lo(.CPI_main_0)], %o0
  for constant pool access.

* We can fold small constant offsets into the %hi/%lo references to constant
  pool addresses as well.

* Directly support select instructions, and fold setcc instructions into them
  where possible.  I think this is what afflicts the inner loop of Olden/tsp
  (hot block = tsp():no_exit.1.i, overall GCC/LLC = 0.03).

$Date$

