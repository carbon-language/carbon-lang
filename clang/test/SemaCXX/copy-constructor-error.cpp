// RUN: clang-cc -fsyntax-only -verify %s 

struct S {
   S (S);  // expected-error {{copy constructor must pass its first argument by reference}}
};

S f();

void g() { 
  S a( f() );  // expected-error {{no matching constructor}}
}
