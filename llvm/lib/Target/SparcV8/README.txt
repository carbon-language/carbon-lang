
SparcV8 backend skeleton
------------------------

This directory houses a 32-bit SPARC V8 backend employing a expander-based
instruction selector.  It is not yet functionally complete.  Watch
this space for more news coming soon!

Current expected test failures
------------------------------

SingleSource/Benchmarks (excluding C++ tests): 
fldry heapsort objinst Queens chomp misr pi whetstone

SingleSource/UnitTests:
C++Catch SimpleC++Test 2003-05-07-VarArgs 2003-07-09-SignedArgs
2003-08-11-VaListArg

To-do
-----

* support calling functions with more than 6 args
* support 64-bit integer (long, ulong) arguments to functions
  - use libc procedures instead of open-coding for:
    __div64 __mul64 __rem64 __udiv64 __umul64 __urem64
* support setcc on longs
* support basic binary operations on longs
* support casting <=32-bit integers, bools to long
* support casting 64-bit integers to FP types
* support varargs intrinsics (va_start et al.)

$Date$

