// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

template<typename T, T val> struct A {};

template<typename T, typename U> constexpr bool is_same = false; // expected-note +{{here}}
template<typename T> constexpr bool is_same<T, T> = true;

namespace String {
  A<const char*, "test"> a; // expected-error {{does not refer to any declaration}}
  A<const char (&)[5], "test"> b; // expected-error {{does not refer to any declaration}}
}

namespace Array {
  char arr[3];
  char x;
  A<const char*, arr> a;
  A<const char(&)[3], arr> b;
  A<const char*, &arr[0]> c;
  A<const char*, &arr[1]> d; // expected-error {{refers to subobject '&arr[1]'}}
  A<const char*, (&arr)[0]> e;
  A<const char*, &x> f;
  A<const char*, &(&x)[0]> g;
  A<const char*, &(&x)[1]> h; // expected-error {{refers to subobject '&x + 1'}}
  A<const char*, 0> i; // expected-error {{not allowed in a converted constant}}
  A<const char*, nullptr> j;
}

namespace Function {
  void f();
  void g() noexcept;
  void h();
  void h(int);
  template<typename...T> void i(T...);
  typedef A<void (*)(), f> a;
  typedef A<void (*)(), &f> a;
  typedef A<void (*)(), g> b;
  typedef A<void (*)(), &g> b;
  typedef A<void (*)(), h> c;
  typedef A<void (*)(), &h> c;
  typedef A<void (*)(), i> d;
  typedef A<void (*)(), &i> d;
  typedef A<void (*)(), i<>> d;
  typedef A<void (*)(), i<int>> e; // expected-error {{is not implicitly convertible}}

  typedef A<void (*)(), 0> x; // expected-error {{not allowed in a converted constant}}
  typedef A<void (*)(), nullptr> y;
}

void Func() {
  A<const char*, __func__> a; // expected-error {{does not refer to any declaration}}
}

namespace LabelAddrDiff {
  void f() {
    a: b: A<int, __builtin_constant_p(true) ? (__INTPTR_TYPE__)&&b - (__INTPTR_TYPE__)&&a : 0> s; // expected-error {{label address difference}}
  };
}

namespace Temp {
  struct S { int n; };
  constexpr S &addr(S &&s) { return s; }
  A<S &, addr({})> a; // expected-error {{constant}} expected-note 2{{temporary}}
  A<S *, &addr({})> b; // expected-error {{constant}} expected-note 2{{temporary}}
  A<int &, addr({}).n> c; // expected-error {{constant}} expected-note 2{{temporary}}
  A<int *, &addr({}).n> d; // expected-error {{constant}} expected-note 2{{temporary}}
}

namespace std { struct type_info; }

namespace RTTI {
  A<const std::type_info&, typeid(int)> a; // expected-error {{does not refer to any declaration}}
  A<const std::type_info*, &typeid(int)> b; // expected-error {{does not refer to any declaration}}
}

namespace PtrMem {
  struct B { int b; };
  struct C : B {};
  struct D : B {};
  struct E : C, D { int e; };

  constexpr int B::*b = &B::b;
  constexpr int C::*cb = b;
  constexpr int D::*db = b;
  constexpr int E::*ecb = cb; // expected-note +{{here}}
  constexpr int E::*edb = db; // expected-note +{{here}}

  constexpr int E::*e = &E::e;
  constexpr int D::*de = (int D::*)e;
  constexpr int C::*ce = (int C::*)e;
  constexpr int B::*bde = (int B::*)de; // expected-note +{{here}}
  constexpr int B::*bce = (int B::*)ce; // expected-note +{{here}}

  // FIXME: This should all be accepted, but we don't yet have a representation
  // nor mangling for this form of template argument.
  using Ab = A<int B::*, b>;
  using Ab = A<int B::*, &B::b>;
  using Abce = A<int B::*, bce>; // expected-error {{not supported}}
  using Abde = A<int B::*, bde>; // expected-error {{not supported}}
  static_assert(!is_same<Ab, Abce>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Ab, Abde>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Abce, Abde>, ""); // expected-error 2{{undeclared}} expected-error {{must be a type}}
  static_assert(is_same<Abce, A<int B::*, (int B::*)(int C::*)&E::e>, ""); // expected-error {{undeclared}} expected-error {{not supported}}

  using Ae = A<int E::*, e>;
  using Ae = A<int E::*, &E::e>;
  using Aecb = A<int E::*, ecb>; // expected-error {{not supported}}
  using Aedb = A<int E::*, edb>; // expected-error {{not supported}}
  static_assert(!is_same<Ae, Aecb>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Ae, Aedb>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Aecb, Aedb>, ""); // expected-error 2{{undeclared}} expected-error {{must be a type}}
  static_assert(is_same<Aecb, A<int E::*, (int E::*)(int C::*)&B::b>, ""); // expected-error {{undeclared}} expected-error {{not supported}}

  using An = A<int E::*, nullptr>;
  using A0 = A<int E::*, (int E::*)0>;
  static_assert(is_same<An, A0>);
}

namespace DeduceDifferentType {
  template<int N> struct A {};
  template<long N> int a(A<N>); // expected-note {{does not have the same type}}
  int a_imp = a(A<3>()); // expected-error {{no matching function}}
  int a_exp = a<3>(A<3>());

  template<decltype(nullptr)> struct B {};
  template<int *P> int b(B<P>); // expected-note {{could not match}} expected-note {{not implicitly convertible}}
  int b_imp = b(B<nullptr>()); // expected-error {{no matching function}}
  int b_exp = b<nullptr>(B<nullptr>()); // expected-error {{no matching function}}

  struct X { constexpr operator int() { return 0; } } x;
  template<X &> struct C {};
  template<int N> int c(C<N>); // expected-note {{does not have the same type}} expected-note {{not implicitly convertible}}
  int c_imp = c(C<x>()); // expected-error {{no matching function}}
  int c_exp = c<x>(C<x>()); // expected-error {{no matching function}}

  struct Z;
  struct Y { constexpr operator Z&(); } y;
  struct Z { constexpr operator Y&() { return y; } } z;
  constexpr Y::operator Z&() { return z; }
  template<Y &> struct D {};
  template<Z &z> int d(D<z>); // expected-note {{does not have the same type}}
  int d_imp = d(D<y>()); // expected-error {{no matching function}}
  int d_exp = d<y>(D<y>());
}

namespace DeclMatch {
  template<typename T, T> int f();
  template<typename T> class X { friend int f<T, 0>(); static int n; };
  template<typename T, T> int f() { return X<T>::n; }
  int k = f<int, 0>(); // ok, friend
}
