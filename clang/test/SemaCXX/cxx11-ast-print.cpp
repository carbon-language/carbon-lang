// RUN: %clang_cc1 -std=c++11 -ast-print %s | FileCheck %s

// FIXME: Print the trailing-return-type properly.
// CHECK: decltype(nullptr) operator "" _foo(const char *p, decltype(sizeof(int)));
auto operator"" _foo(const char *p, decltype(sizeof(int))) -> decltype(nullptr);

// CHECK: const char *p1 = "bar1"_foo;
const char *p1 = "bar1"_foo;
// CHECK: const char *p2 = "bar2"_foo;
const char *p2 = R"x(bar2)x"_foo;
// CHECK: const char *p3 = u8"bar3"_foo;
const char *p3 = u8"bar3"_foo;
