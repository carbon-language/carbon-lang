
SparcV8 backend skeleton
------------------------

This directory houses a 32-bit SPARC V8 backend employing an expander-based
instruction selector.  It is not yet functionally complete.  Watch
this space for more news coming soon!

Current expected test failures
------------------------------

All SingleSource/Benchmarks tests are expected to pass.  Currently, all
C++ tests are expected to fail.  Here are the known SingleSource failures:

	UnitTests/SetjmpLongjmp/C++/C++Catch
	UnitTests/SetjmpLongjmp/C++/SimpleC++Test
	Regression/C++/EH/ConditionalExpr
	Regression/C++/EH/ctor_dtor_count-2
	Regression/C++/EH/ctor_dtor_count
	Regression/C++/EH/exception_spec_test
	Regression/C++/EH/function_try_block
	Regression/C++/EH/simple_rethrow
	Regression/C++/EH/simple_throw
	Regression/C++/EH/throw_rethrow_test
	CustomChecked/oopack_v1p8

To-do
-----

* support setcc on longs
* support shifts on longs
* support casting 64-bit integers to FP types
* support FP rem

$Date$

