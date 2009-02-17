// RUN: clang -fsyntax-only -verify %s
template<typename T, typename U = int> class A;

template<> class A<double, double>; // expected-note{{forward declaration}}

template<> class A<float, float> {  // expected-note{{previous definition}}
  int x;
};

template<> class A<float> { // expected-note{{previous definition}}
  int y;
};

int test_specs(A<float, float> *a1, A<float, int> *a2) {
  return a1->x + a2->y;
}

int test_incomplete_specs(A<double, double> *a1, 
                          A<double> *a2) // FIXME: expected-note{{forward declaration}}
{
  (void)a1->x; // expected-error{{incomplete definition of type 'A<double, double>'}}
  (void)a2->x; // expected-error{{incomplete definition of type 'A<double>'}}
}

typedef float FLOAT;

template<> class A<float, FLOAT>;

template<> class A<FLOAT, float> { }; // expected-error{{redefinition}}

template<> class A<float, int> { }; // expected-error{{redefinition}}
