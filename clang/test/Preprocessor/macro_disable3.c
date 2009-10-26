// RUN: clang-cc %s -E | FileCheck %s
// Check for C99 6.10.3.4p2.

#define f(a) f(x * (a)) 
#define x 2 
#define z z[0] 
f(f(z)); 

// CHECK: f(2 * (f(2 * (z[0]))));

