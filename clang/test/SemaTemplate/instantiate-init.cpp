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

// Instantiate out-of-line definitions of static data members which complete
// types through an initializer even when the only use of the member that would
// cause instantiation is in an unevaluated context, but one requiring its
// complete type.
namespace PR10001 {
  template <typename T> struct S {
    static const int arr[];
    static const int x;
    static int f();
  };

  template <typename T> const int S<T>::arr[] = { 1, 2, 3 };
  template <typename T> const int S<T>::x = sizeof(arr) / sizeof(arr[0]);
  template <typename T> int S<T>::f() { return x; }

  int x = S<int>::f();
}

namespace PR7985 {
  template<int N> struct integral_c { };

  template <typename T, int N>
  integral_c<N> array_lengthof(T (&x)[N]) { return integral_c<N>(); }

  struct Data {
    int x;
  };

  template<typename T>
  struct Description {
    static const Data data[];
  };

  template<typename T>
  const Data Description<T>::data[] = {{ 0 }};

  void test() {
    integral_c<1> ic1 = array_lengthof(Description<int>::data);
  }
}
