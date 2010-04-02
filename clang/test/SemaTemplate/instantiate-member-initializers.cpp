// RUN: %clang_cc1 -fsyntax-only -Wall -verify %s

template<typename T> struct A {
  A() : a(1) { } // expected-error{{cannot initialize a member subobject of type 'void *' with an rvalue of type 'int'}}

  T a;
};

A<int> a0;
A<void*> a1; // expected-note{{in instantiation of member function 'A<void *>::A' requested here}}

template<typename T> struct B {
  B() : b(1), // expected-warning {{member 'b' will be initialized after}}
    a(2) { } // expected-note {{field a}}
  
  int a;
  int b;
};

B<int> b0; // expected-note {{in instantiation of member function 'B<int>::B' requested here}}

template <class T> struct AA { AA(int); };
template <class T> class BB : public AA<T> {
  BB() : AA<T>(1) {}
};
BB<int> x;
