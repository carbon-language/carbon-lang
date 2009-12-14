// RUN: clang-cc %s -E | FileCheck %s

#define foo(x) bar x
foo(foo) (2)
// CHECK: bar foo (2)

#define m(a) a(w)
#define w ABCD
m(m)
// CHECK: m(ABCD)
