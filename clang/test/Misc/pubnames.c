// RUN: %clang_cc1 -pubnames-dump %s | FileCheck %s
#define FOO
#define BAR
#undef FOO
#define WIBBLE

int foo();
int bar(float);
int wibble;

// CHECK: BAR
// CHECK-NOT: FOO
// CHECK: WIBBLE
// CHECK-NOT: __clang_major__
// CHECK: bar
// CHECK: foo
// CHECK: wibble


