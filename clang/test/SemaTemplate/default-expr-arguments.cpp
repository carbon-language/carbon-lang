// RUN: clang-cc -fsyntax-only -verify %s

struct S { };

template<typename T> void f1(T a, T b = 10) { } // expected-error{{cannot initialize 'b' with an rvalue of type 'int'}}

template<typename T> void f2(T a, T b = T()) { }

template<typename T> void f3(T a, T b = T() + T()); // expected-error{{invalid operands to binary expression ('struct S' and 'struct S')}}

void g() {
  f1(10);
  f1(S()); // expected-note{{in instantiation of default function argument expression for 'f1<struct S>' required here}}
  
  f2(10);
  f2(S());
  
  f3(10);
  f3(S()); // expected-note{{in instantiation of default function argument expression for 'f3<struct S>' required here}}
}

template<typename T> struct F {
  F(T t = 10);
};

struct FD : F<int> { };

void g2() {
  F<int> f;
  FD fd;
}

template<typename T> struct G {
  G(T) {}
};

void s(G<int> flags = 10) { }


