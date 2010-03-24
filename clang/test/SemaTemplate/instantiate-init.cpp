// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X0 { // expected-note 4{{candidate}}
  X0(int*, float*); // expected-note 4{{candidate}}
};

template<typename T, typename U>
X0 f0(T t, U u) {
  X0 x0(t, u); // expected-error{{no matching}}
  return X0(t, u); // expected-error{{no matching}}
}

void test_f0(int *ip, float *fp, double *dp) {
  f0(ip, fp);
  f0(ip, dp); // expected-note{{instantiation}}
}

template<typename Ret, typename T, typename U>
Ret f1(Ret *retty, T t, U u) {
  Ret r0(t, u); // expected-error{{no matching}}
  return Ret(t, u); // expected-error{{no matching}}
}

void test_f1(X0 *x0, int *ip, float *fp, double *dp) {
  f1(x0, ip, fp);
  f1(x0, ip, dp); // expected-note{{instantiation}}
}

namespace PR6457 {
  template <typename T> struct X { explicit X(T* p = 0) { }; };
  template <typename T> struct Y { Y(int, const T& x); };
  struct A { };
  template <typename T>
  struct B {
    B() : y(0, X<A>()) { }
    Y<X<A> > y;
  };
  B<int> b;
}

namespace PR6657 {
  struct X
  {
    X (int, int) { }
  };

  template <typename>
  void f0()
  {
    X x = X(0, 0);
  }

  void f1()
  {
    f0<int>();
  }
}
