
SparcV8 backend skeleton
------------------------

This directory houses a 32-bit SPARC V8 backend employing a expander-based
instruction selector.  It is not yet functionally complete.  Watch
this space for more news coming soon!

Current expected test failures
------------------------------

All SingleSource/Benchmarks tests are expected to pass.  Currently, all
C++ tests and all tests involving varargs intrinsics (use of
va_start/va_end) are expected to fail.  Here are the known SingleSource
failures:

	UnitTests/SetjmpLongjmp/C++/C++Catch
	UnitTests/SetjmpLongjmp/C++/SimpleC++Test
	UnitTests/2003-05-07-VarArgs
	UnitTests/2003-07-09-SignedArgs
	UnitTests/2003-08-11-VaListArg
	Regression/C++/EH/ConditionalExpr
	Regression/C++/EH/ctor_dtor_count-2
	Regression/C++/EH/ctor_dtor_count
	Regression/C++/EH/exception_spec_test
	Regression/C++/EH/function_try_block
	Regression/C++/EH/simple_rethrow
	Regression/C++/EH/simple_throw
	Regression/C++/EH/throw_rethrow_test
	Regression/C/casts
	CustomChecked/oopack_v1p8

To-do
-----

* support setcc on longs
* support basic binary operations on longs
  - use libc procedures instead of open-coding for:
    __div64 __mul64 __rem64 __udiv64 __umul64 __urem64
* support casting 64-bit integers to FP types
* support varargs intrinsics (va_start et al.)

$Date$

