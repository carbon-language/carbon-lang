// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__launch_bounds__(128, 7) void Test1(void);
__launch_bounds__(128) void Test2(void);

__launch_bounds__(1, 2, 3) void Test3(void); // expected-error {{'launch_bounds' attribute takes no more than 2 arguments}}

// FIXME: the error should read that the attribute takes exactly one or two arguments, but there
// is no support for such a diagnostic currently.
__launch_bounds__() void Test4(void); // expected-error {{'launch_bounds' attribute takes no more than 2 arguments}}

int Test5 __launch_bounds__(128, 7); // expected-warning {{'launch_bounds' attribute only applies to functions and methods}}
