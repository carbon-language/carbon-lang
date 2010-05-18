// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X1 {
  friend void f6(int) { } // expected-error{{redefinition of}} \
                          // expected-note{{previous definition}}
};

X1<int> x1a; 
X1<float> x1b; // expected-note {{in instantiation of}}
