// RUN: clang-cc -fsyntax-only -verify %s 

struct S { // expected-note {{candidate function}} 
   S (S);  // expected-error {{copy constructor must pass its first argument by reference}} \\
           // expected-note {{candidate function}}
};

S f();

void g() { 
  S a( f() );  // expected-error {{call to constructor of 'a' is ambiguous}}
}

