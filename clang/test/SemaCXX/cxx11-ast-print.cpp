// RUN: %clang_cc1 -std=c++11 -ast-print %s | FileCheck %s

// FIXME: Print the trailing-return-type properly.
// CHECK: decltype(nullptr) operator "" _foo(const char *p, decltype(sizeof(int)));
auto operator"" _foo(const char *p, decltype(sizeof(int))) -> decltype(nullptr);

// CHECK: decltype(""_foo) operator "" _bar(unsigned long long);
decltype(""_foo) operator"" _bar(unsigned long long);

// CHECK: decltype(42_bar) operator "" _baz(long double);
decltype(42_bar) operator"" _baz(long double);

// CHECK: const char *p1 = "bar1"_foo;
const char *p1 = "bar1"_foo;
// CHECK: const char *p2 = "bar2"_foo;
const char *p2 = R"x(bar2)x"_foo;
// CHECK: const char *p3 = u8"bar3"_foo;
const char *p3 = u8"bar3"_foo;
// CHECK: const char *p4 = 297_bar;
const char *p4 = 0x129_bar;
// CHECK: const char *p5 = 1.0E+12_baz;
const char *p5 = 1e12_baz;
