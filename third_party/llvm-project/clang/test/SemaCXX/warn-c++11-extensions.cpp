// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wc++11-extensions -verify %s

long long ll1 = // expected-warning {{'long long' is a C++11 extension}}
         -42LL; // expected-warning {{'long long' is a C++11 extension}}
unsigned long long ull1 = // expected-warning {{'long long' is a C++11 extension}}
                   42ULL; // expected-warning {{'long long' is a C++11 extension}}

enum struct E1 { A, B }; // expected-warning {{scoped enumerations are a C++11 extension}}
enum class E2 { C, D }; // expected-warning {{scoped enumerations are a C++11 extension}}
