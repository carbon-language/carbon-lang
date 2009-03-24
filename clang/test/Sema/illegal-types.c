// RUN: clang-cc -fsyntax-only -verify -std=c++98 %s

void a (void []()); // expected-error{{'type name' declared as array of functions}}
void b (void p[]()); // expected-error{{'p' declared as array of functions}}
void c (int &[]); // expected-error{{'type name' declared as array of references}}
void d (int &p[]); // expected-error{{'p' declared as array of references}}

