
SparcV8 backend skeleton
------------------------

This directory houses a 32-bit SPARC V8 backend employing an expander-based
instruction selector.  It is not yet functionally complete.  Watch
this space for more news coming soon!

Current expected test failures
------------------------------

The SparcV8 backend works on many simple C++ SingleSource codes. Here
are the known SingleSource failures:

	UnitTests/SetjmpLongjmp/C++/SimpleC++Test
	Regression/C++/EH/exception_spec_test
	Regression/C++/EH/throw_rethrow_test
	Benchmarks/Shootout-C++/moments
	Benchmarks/Shootout-C++/random

Here are some known MultiSource test failures - this is probably not a
complete list right now.

	burg siod lambda make_dparser hbd treecc hexxagon fhourstones
	bisect testtrie eks imp bh power anagram bc distray

To-do
-----

* support shifts on longs
* support casting 64-bit integers to FP types
* support FP rem

$Date$

