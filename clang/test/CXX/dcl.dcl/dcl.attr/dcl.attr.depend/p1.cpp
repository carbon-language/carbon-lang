// RUN: %clang_cc1 -verify -std=c++11 %s

[[carries_dependency, carries_dependency]] int m1(); // expected-error {{attribute 'carries_dependency' cannot appear multiple times in an attribute specifier}}
[[carries_dependency]] [[carries_dependency]] int m2(); // ok
[[carries_dependency()]] int m3(); // expected-error {{attribute 'carries_dependency' cannot have an argument list}}

[[carries_dependency]] void f1(); // FIXME: warn here
[[carries_dependency]] int f2(); // ok
int f3(int param [[carries_dependency]]); // ok
[[carries_dependency]] int (*f4)(); // expected-error {{'carries_dependency' attribute only applies to functions, methods, and parameters}}
int (*f5 [[carries_dependency]])(); // expected-error {{'carries_dependency' attribute only applies to functions, methods, and parameters}}
int (*f6)() [[carries_dependency]]; // expected-error {{'carries_dependency' attribute cannot be applied to types}}

auto l1 = [](int n [[carries_dependency]]) {};
// There's no way to write a lambda such that the return value carries
// a dependency, because an attribute applied to the lambda appertains to
// the *type* of the operator() function, not to the function itself.
auto l2 = []() [[carries_dependency]] {}; // expected-error {{'carries_dependency' attribute cannot be applied to types}}
