// RUN: %clang_cc1 -fsyntax-only -verify %s

template void *; // expected-error{{expected unqualified-id}}

template typedef void f0; // expected-error{{explicit instantiation of typedef}}

int v0; // expected-note{{refers here}}
template int v0; // expected-error{{does not refer}}

template<typename T>
struct X0 {
  static T value;
  
  T f0(T x) {
    return x + 1;  // expected-error{{invalid operands}}
  } 
  T* f0(T*, T*) { return T(); }
  
  template<typename U>
  T f0(T, U) { return T(); }
};

template<typename T>
T X0<T>::value; // expected-error{{no matching constructor}}

template int X0<int>::value;

struct NotDefaultConstructible { // expected-note{{candidate constructor (the implicit copy constructor)}}
  NotDefaultConstructible(int); // expected-note{{candidate constructor}}
};

template NotDefaultConstructible X0<NotDefaultConstructible>::value; // expected-note{{instantiation}}

template int X0<int>::f0(int);
template int* X0<int>::f0(int*, int*);
template int X0<int>::f0(int, float);

template int X0<int>::f0(int) const; // expected-error{{does not refer}}
template int* X0<int>::f0(int*, float*); // expected-error{{does not refer}}

struct X1 { };
typedef int X1::*MemPtr;

template MemPtr X0<MemPtr>::f0(MemPtr); // expected-note{{requested here}}

struct X2 {
  int f0(int); // expected-note{{refers here}}
  
  template<typename T> T f1(T) { return T(); }
  template<typename T> T* f1(T*) { return 0; }

  template<typename T, typename U> void f2(T, U*) { } // expected-note{{candidate}}
  template<typename T, typename U> void f2(T*, U) { } // expected-note{{candidate}}
};

template int X2::f0(int); // expected-error{{not an instantiation}}

template int *X2::f1(int *); // okay

template void X2::f2(int *, int *); // expected-error{{ambiguous}}


template<typename T> void print_type() { }

template void print_type<int>();
template void print_type<float>();

template<typename T> void print_type(T*) { }

template void print_type(int*);
template void print_type<int>(float*); // expected-error{{does not refer}}

void print_type(double*);
template void print_type<double>(double*);

// PR5069
template<int I> void foo0 (int (&)[I + 1]) { }
template void foo0<2> (int (&)[3]);
