// RUN: %clang_cc1 -E -dM %s | FileCheck --check-prefix=CHECK-GNU-COMPAT %s
// RUN: %clang_cc1 -std=c++98 -E -dM %s | FileCheck --check-prefix=CHECK-CONFORMING %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -verify -Weverything %s
#include <stdbool.h>
#define zzz

// CHECK-GNU-COMPAT: #define _Bool bool
// CHECK-GNU-COMPAT: #define bool bool
// CHECK-GNU-COMPAT: #define false false
// CHECK-GNU-COMPAT: #define true true

// CHECK-CONFORMING-NOT: #define _Bool
// CHECK-CONFORMING: #define __CHAR_BIT__
// CHECK-CONFORMING-NOT: #define false false
// CHECK-CONFORMING: #define zzz

zzz
// expected-no-diagnostics
extern bool x;
