// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  static T value;
};

template<typename T>
T X0<T>::value = 0; // expected-error{{initialize}}

struct X1 { 
  X1(int);
};

struct X2 { };

int& get_int() { return X0<int>::value; }
X1& get_X1() { return X0<X1>::value; }

double*& get_double_ptr() { return X0<int*>::value; } // expected-error{{initialized}}

X2& get_X2() { 
  return X0<X2>::value; // expected-note{{instantiation}}
}
  
template<typename T> T x; // expected-error{{variable 'x' declared as a template}}
