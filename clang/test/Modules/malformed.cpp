// This test explicitly cd's to the test/Modules directory so that we can test
// that filenames found via relative -I paths are printed correctly.
//
// REQUIRES: shell
//
// RUN: rm -rf %t
// RUN: cd %S
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I Inputs/malformed -DHEADER="a1.h" %s 2>&1 | FileCheck %s --check-prefix=CHECK-A
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I Inputs/malformed -DHEADER="b1.h" %s 2>&1 | FileCheck %s --check-prefix=CHECK-B
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t -I Inputs/malformed -DHEADER="c.h" malformed.cpp 2>&1 | FileCheck %s --check-prefix=CHECK-C

#define STR2(x) #x
#define STR(x) STR2(x)
#include STR(HEADER)

// CHECK-A: While building module 'malformed_a'
// CHECK-A: {{^}}Inputs/malformed/a1.h:1:{{.*}} error: expected '}'
// CHECK-A: {{^}}Inputs/malformed/a1.h:1:{{.*}} note: to match this '{'
//
// CHECK-A: While building module 'malformed_a'
// CHECK-A: {{^}}Inputs/malformed/a2.h:1:{{.*}} error: extraneous closing brace

// CHECK-B: While building module 'malformed_b'
// CHECK-B: {{^}}Inputs/malformed/b1.h:2:{{.*}} error: expected '}'
// CHECK-B: {{^}}Inputs/malformed/b1.h:1:{{.*}} note: to match this '{'
// CHECK-B: {{^}}Inputs/malformed/b1.h:3:{{.*}} error: extraneous closing brace ('}')
//
// CHECK-B: While building module 'malformed_b'
// CHECK-B: {{^}}Inputs/malformed/b2.h:1:{{.*}} error: redefinition of 'g'
// CHECK-B: {{^}}Inputs/malformed/b2.h:1:{{.*}} note: previous definition is here

void test() { f<int>(); }
// Test that we use relative paths to name files within an imported module.
//
// CHECK-C: In module 'c' imported from malformed.cpp:14:
// CHECK-C: {{^}}Inputs/malformed/c.h:1:33: error: type 'int' cannot be used prior to '::'
// CHECK-C: {{^}}malformed.cpp:[[@LINE-5]]:15: note: in instantiation of
