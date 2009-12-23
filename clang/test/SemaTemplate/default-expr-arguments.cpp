// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
class C { C(int a0 = 0); };

template<>
C<char>::C(int a0);

struct S { }; // expected-note 3 {{candidate function}}

template<typename T> void f1(T a, T b = 10) { } // expected-error{{no viable conversion}}

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
  F(T t = 10); // expected-error{{no viable conversion}}
  void f(T t = 10); // expected-error{{no viable conversion}}
};

struct FD : F<int> { };

void g2() {
  F<int> f;
  FD fd;
}

void g3(F<int> f, F<struct S> s) {
  f.f();
  s.f(); // expected-note{{in instantiation of default function argument expression for 'f<struct S>' required here}}
  
  F<int> f2;
  F<S> s2; // expected-note{{in instantiation of default function argument expression for 'F<struct S>' required here}}
}

template<typename T> struct G {
  G(T) {}
};

void s(G<int> flags = 10) { }

// Test default arguments
template<typename T>
struct X0 {
  void f(T = T()); // expected-error{{no matching}}
};

template<typename U>
void X0<U>::f(U) { }

void test_x0(X0<int> xi) {
  xi.f();
  xi.f(17);
}

struct NotDefaultConstructible { // expected-note 2{{candidate}}
  NotDefaultConstructible(int); // expected-note 2{{candidate}}
};

void test_x0_not_default_constructible(X0<NotDefaultConstructible> xn) {
  xn.f(NotDefaultConstructible(17));
  xn.f(42);
  xn.f(); // expected-note{{in instantiation of default function argument}}
}

template<typename T>
struct X1 {
  typedef T value_type;
  X1(const value_type& value = value_type());
};

void test_X1() {
  X1<int> x1;
}

template<typename T>
struct X2 {
  void operator()(T = T()); // expected-error{{no matching}}
};

void test_x2(X2<int> x2i, X2<NotDefaultConstructible> x2n) {
  x2i();
  x2i(17);
  x2n(NotDefaultConstructible(17));
  x2n(); // expected-note{{in instantiation of default function argument}}
}

// PR5283
namespace PR5283 {
template<typename T> struct A {
  A(T = 1); // expected-error 3 {{cannot initialize a parameter of type 'int *' with an rvalue of type 'int'}}
};

struct B : A<int*> { 
  B();
};
B::B() { } // expected-note {{in instantiation of default function argument expression for 'A<int *>' required he}}

struct C : virtual A<int*> {
  C();
};
C::C() { } // expected-note {{in instantiation of default function argument expression for 'A<int *>' required he}}

struct D {
  D();
  
  A<int*> a;
};
D::D() { } // expected-note {{in instantiation of default function argument expression for 'A<int *>' required he}}
}

// PR5301
namespace pr5301 {
  void f(int, int = 0);

  template <typename T>
  void g(T, T = 0);

  template <int I>
  void i(int a = I);

  template <typename T>
  void h(T t) {
    f(0);
    g(1);
    g(t);
    i<2>();
  }

  void test() {
    h(0);
  }
}

// PR5810
namespace PR5810 {
  template<typename T>
  struct allocator {
    allocator() { int a[sizeof(T) ? -1 : -1]; } // expected-error{{array size is negative}}
  };
  
  template<typename T>
  struct vector {
    vector(const allocator<T>& = allocator<T>()) {} // expected-note{{instantiation of}}
  };
  
  struct A { };
  
  template<typename>
  void FilterVTs() {
    vector<A> Result;
  }
  
  void f() {
    vector<A> Result;
  }
}
