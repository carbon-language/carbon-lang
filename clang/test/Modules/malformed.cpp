// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/malformed -DHEADER="a1.h" %s 2>&1 | FileCheck %s --check-prefix=CHECK-A
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/malformed -DHEADER="b1.h" %s 2>&1 | FileCheck %s --check-prefix=CHECK-B

#define STR2(x) #x
#define STR(x) STR2(x)
#include STR(HEADER)

// CHECK-A: While building module 'malformed_a'
// CHECK-A: a1.h:1:{{.*}} error: expected '}'
// CHECK-A: a1.h:1:{{.*}} note: to match this '{'
//
// CHECK-A: While building module 'malformed_a'
// CHECK-A: a2.h:1:{{.*}} error: extraneous closing brace

// CHECK-B: While building module 'malformed_b'
// CHECK-B: b1.h:2:{{.*}} error: expected '}'
// CHECK-B: b1.h:1:{{.*}} note: to match this '{'
// CHECK-B: b1.h:3:{{.*}} error: extraneous closing brace ('}')
//
// CHECK-B: While building module 'malformed_b'
// CHECK-B: b2.h:1:{{.*}} error: redefinition of 'g'
// CHECK-B: b2.h:1:{{.*}} note: previous definition is here
