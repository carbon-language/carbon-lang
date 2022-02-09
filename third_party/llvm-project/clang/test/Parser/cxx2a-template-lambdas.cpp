// RUN: %clang_cc1 -std=c++2b %s -verify
// RUN: %clang_cc1 -std=c++2a %s -verify

auto L0 = []<> { }; //expected-error {{cannot be empty}}

auto L1 = []<typename T1, typename T2> { };
auto L2 = []<typename T1, typename T2>(T1 arg1, T2 arg2) -> T1 { };
auto L3 = []<typename T>(auto arg) { T t; };
auto L4 = []<int I>() { };

// http://llvm.org/PR49736
auto L5 = []<auto>(){};
auto L6 = []<auto>{};
auto L7 = []<auto>() noexcept {};
auto L8 = []<auto> noexcept {};
#if __cplusplus <= 202002L
// expected-warning@-2 {{lambda without a parameter clause is a C++2b extension}}
#endif
auto L9 = []<auto> requires true {};
auto L10 = []<auto> requires true(){};
auto L11 = []<auto> requires true() noexcept {};
auto L12 = []<auto> requires true noexcept {};
#if __cplusplus <= 202002L
// expected-warning@-2 {{is a C++2b extension}}
#endif
auto L13 = []<auto>() noexcept requires true {};
auto L14 = []<auto> requires true() noexcept requires true {};

auto XL0 = []<auto> noexcept requires true {};               // expected-error {{expected body of lambda expression}}
auto XL1 = []<auto> requires true noexcept requires true {}; // expected-error {{expected body}}
#if __cplusplus <= 202002L
// expected-warning@-3 {{is a C++2b extension}}
// expected-warning@-3 {{is a C++2b extension}}
#endif
