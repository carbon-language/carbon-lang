// RUN: %clang_cc1 -fsyntax-only -verify %s
template<class T> struct A {
  static T t; // expected-error{{static data member instantiated with function type 'int ()'}}
};
typedef int function();
A<function> a; // expected-note{{instantiation of}}

template<typename T> struct B {
  B() { T t; } // expected-error{{variable instantiated with function type 'int ()'}}
};
B<function> b; // expected-note{{instantiation of}}

template <typename T> int f0(void *, const T&); // expected-note{{candidate template ignored: substitution failure}}
enum {e};

void test_f0(int n) {
  int i = f0(0, e); // FIXME: We should get a warning here, at least
  int vla[n];
  f0(0, vla); // expected-error{{no matching function for call to 'f0'}}
}

