// RUN: clang-cc -fsyntax-only -verify %s
template<typename T, typename U = float> struct A { };

typedef A<int> A_int;

typedef float FLOAT;

A<int, FLOAT> *foo(A<int> *ptr, A<int> const *ptr2, A<int, double> *ptr3) {
  if (ptr)
    return ptr; // okay
  else if (ptr2)
    return ptr2; // expected-error{{incompatible type returning 'A<int> const *', expected 'A<int, FLOAT> *'}}
  else {
    return ptr3; // expected-error{{incompatible type returning 'A<int, double> *', expected 'A<int, FLOAT> *'}}
  }
}

template<int I> struct B;

const int value = 12;
B<17 + 2> *bar(B<(19)> *ptr1, B< (::value + 7) > *ptr2, B<19 - 3> *ptr3) {
  if (ptr1)
    return ptr1;
  else if (ptr2)
    return ptr2;
  else
    return ptr3; // expected-error{{incompatible type returning 'B<19 - 3> *', expected 'B<17 + 2> *'}}
}

typedef B<5> B5;


namespace N {
  template<typename T> struct C {};
}

N::C<int> c1;
typedef N::C<float> c2;

// PR5655
template<typename T> struct Foo { }; // expected-note{{template is declared here}}

void f(void) { Foo bar; } // expected-error{{without a template argument list}}
