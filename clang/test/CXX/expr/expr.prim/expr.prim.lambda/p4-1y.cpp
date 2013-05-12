// RUN: %clang_cc1 -fsyntax-only -std=c++1y %s -verify

int a;
int &b = [] (int &r) -> decltype(auto) { return r; } (a);
int &c = [] (int &r) -> decltype(auto) { return (r); } (a);
int &d = [] (int &r) -> auto & { return r; } (a);
int &e = [] (int &r) -> auto { return r; } (a); // expected-error {{cannot bind to a temporary}}
int &f = [] (int r) -> decltype(auto) { return r; } (a); // expected-error {{cannot bind to a temporary}}
int &g = [] (int r) -> decltype(auto) { return (r); } (a); // expected-warning {{reference to stack}}
