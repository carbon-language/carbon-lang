// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

template<class X, class Y, class Z>
class A {};
template<class X>
class B {};
template<class X>
class C {};

void foo_abbb(A<B<char>, B<char>, B<char> >) {}
// CHECK: "?foo_abbb@@YAXV?$A@V?$B@D@@V1@V1@@@@Z"
void foo_abb(A<char, B<char>, B<char> >) {}
// CHECK: "?foo_abb@@YAXV?$A@DV?$B@D@@V1@@@@Z"
void foo_abc(A<char, B<char>, C<char> >) {}
// CHECK: "?foo_abc@@YAXV?$A@DV?$B@D@@V?$C@D@@@@@Z"
void foo_bt(bool a, B<bool(bool)> b) {}
// CHECK: "?foo_bt@@YAX_NV?$B@$$A6A_N_N@Z@@@Z"

namespace N {
template<class X, class Y, class Z>
class A {};
template<class X>
class B {};
template<class X>
class C {};
template<class X, class Y>
class D {};
class Z {};
}

void foo_abbb(N::A<N::B<char>, N::B<char>, N::B<char> >) {}
// CHECK: "?foo_abbb@@YAXV?$A@V?$B@D@N@@V12@V12@@N@@@Z"
void foo_abb(N::A<char, N::B<char>, N::B<char> >) {}
// CHECK: "?foo_abb@@YAXV?$A@DV?$B@D@N@@V12@@N@@@Z"
void foo_abc(N::A<char, N::B<char>, N::C<char> >) {}
// CHECK: "?foo_abc@@YAXV?$A@DV?$B@D@N@@V?$C@D@2@@N@@@Z"

N::A<char, N::B<char>, N::C<char> > abc_foo() {
// CHECK: ?abc_foo@@YA?AV?$A@DV?$B@D@N@@V?$C@D@2@@N@@XZ
  return N::A<char, N::B<char>, N::C<char> >();
}

N::Z z_foo(N::Z arg) {
// CHECK: ?z_foo@@YA?AVZ@N@@V12@@Z
  return arg;
}

N::B<char> b_foo(N::B<char> arg) {
// CHECK: ?b_foo@@YA?AV?$B@D@N@@V12@@Z
  return arg;
}

N::D<char, char> d_foo(N::D<char, char> arg) {
// CHECK: ?d_foo@@YA?AV?$D@DD@N@@V12@@Z
  return arg;
}

N::A<char, N::B<char>, N::C<char> > abc_foo_abc(N::A<char, N::B<char>, N::C<char> >) {
// CHECK: ?abc_foo_abc@@YA?AV?$A@DV?$B@D@N@@V?$C@D@2@@N@@V12@@Z
  return N::A<char, N::B<char>, N::C<char> >();
}

namespace NA {
class X {};
template<class T> class Y {};
}

namespace NB {
class X {};
template<class T> class Y {};
}

void foo5(NA::Y<NB::Y<NA::Y<NB::Y<NA::X> > > > arg) {}
// CHECK: "?foo5@@YAXV?$Y@V?$Y@V?$Y@V?$Y@VX@NA@@@NB@@@NA@@@NB@@@NA@@@Z"

void foo11(NA::Y<NA::X>, NB::Y<NA::X>) {}
// CHECK: "?foo11@@YAXV?$Y@VX@NA@@@NA@@V1NB@@@Z"

void foo112(NA::Y<NA::X>, NB::Y<NB::X>) {}
// CHECK: "?foo112@@YAXV?$Y@VX@NA@@@NA@@V?$Y@VX@NB@@@NB@@@Z"

void foo22(NA::Y<NB::Y<NA::X> >, NB::Y<NA::Y<NA::X> >) {}
// CHECK: "?foo22@@YAXV?$Y@V?$Y@VX@NA@@@NB@@@NA@@V?$Y@V?$Y@VX@NA@@@NA@@@NB@@@Z"

namespace PR13207 {
class A {};
class B {};
class C {};

template<class X>
class F {};
template<class X>
class I {};
template<class X, class Y>
class J {};
template<class X, class Y, class Z>
class K {};

class L {
 public:
  void foo(I<A> x) {}
};
// CHECK: "?foo@L@PR13207@@QAEXV?$I@VA@PR13207@@@2@@Z"

void call_l_foo(L* l) { l->foo(I<A>()); }

void foo(I<A> x) {}
// CHECK: "?foo@PR13207@@YAXV?$I@VA@PR13207@@@1@@Z"
void foo2(I<A> x, I<A> y) { }
// CHECK: "?foo2@PR13207@@YAXV?$I@VA@PR13207@@@1@0@Z"
void bar(J<A,B> x) {}
// CHECK: "?bar@PR13207@@YAXV?$J@VA@PR13207@@VB@2@@1@@Z"
void spam(K<A,B,C> x) {}
// CHECK: "?spam@PR13207@@YAXV?$K@VA@PR13207@@VB@2@VC@2@@1@@Z"

void baz(K<char, F<char>, I<char> >) {}
// CHECK: "?baz@PR13207@@YAXV?$K@DV?$F@D@PR13207@@V?$I@D@2@@1@@Z"
void qux(K<char, I<char>, I<char> >) {}
// CHECK: "?qux@PR13207@@YAXV?$K@DV?$I@D@PR13207@@V12@@1@@Z"

namespace NA {
class X {};
template<class T> class Y {};
void foo(Y<X> x) {}
// CHECK: "?foo@NA@PR13207@@YAXV?$Y@VX@NA@PR13207@@@12@@Z"
void foofoo(Y<Y<X> > x) {}
// CHECK: "?foofoo@NA@PR13207@@YAXV?$Y@V?$Y@VX@NA@PR13207@@@NA@PR13207@@@12@@Z"
}

namespace NB {
class X {};
template<class T> class Y {};
void foo(Y<NA::X> x) {}
// CHECK: "?foo@NB@PR13207@@YAXV?$Y@VX@NA@PR13207@@@12@@Z"

void bar(NA::Y<X> x) {}
// CHECK: "?bar@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@NA@2@@Z"

void spam(NA::Y<NA::X> x) {}
// CHECK: "?spam@NB@PR13207@@YAXV?$Y@VX@NA@PR13207@@@NA@2@@Z"

void foobar(NA::Y<Y<X> > a, Y<Y<X> >) {}
// CHECK: "?foobar@NB@PR13207@@YAXV?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V312@@Z"

void foobarspam(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >) {}
// CHECK: "?foobarspam@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@@Z"

void foobarbaz(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >, Y<Y<X> > c) {}
// CHECK: "?foobarbaz@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@2@Z"

void foobarbazqux(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >, Y<Y<X> > c , NA::Y<Y<Y<X> > > d) {}
// CHECK: "?foobarbazqux@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@2V?$Y@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NB@PR13207@@@52@@Z"
}

namespace NC {
class X {};
template<class T> class Y {};

void foo(Y<NB::X> x) {}
// CHECK: "?foo@NC@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@@Z"

void foobar(NC::Y<NB::Y<NA::Y<NA::X> > > x) {}
// CHECK: "?foobar@NC@PR13207@@YAXV?$Y@V?$Y@V?$Y@VX@NA@PR13207@@@NA@PR13207@@@NB@PR13207@@@12@@Z"
}
}

// Function template names are not considered for backreferencing, but normal
// function names are.
namespace fn_space {
struct RetVal { int hash; };
template <typename T>
RetVal fun_tmpl(const T &t) { return RetVal(); }
RetVal fun_normal(int t) { return RetVal(); }
void fun_instantiate() {
  fun_normal(1);
  fun_tmpl(1);
}
// CHECK: "?fun_normal@fn_space@@YA?AURetVal@1@H@Z"
// CHECK: "??$fun_tmpl@H@fn_space@@YA?AURetVal@0@ABH@Z"

template <typename T, RetVal (*F)(T)>
RetVal fun_tmpl_recurse(T t) {
  if (!t)
    return RetVal();
  return F(t - 1);
}
RetVal ident(int x) { return RetVal(); }
void fun_instantiate2() {
  fun_tmpl_recurse<int, fun_tmpl_recurse<int, ident> >(10);
}
// CHECK: "??$fun_tmpl_recurse@H$1??$fun_tmpl_recurse@H$1?ident@fn_space@@YA?AURetVal@2@H@Z@fn_space@@YA?AURetVal@1@H@Z@fn_space@@YA?AURetVal@0@H@Z"
// CHECK: "??$fun_tmpl_recurse@H$1?ident@fn_space@@YA?AURetVal@2@H@Z@fn_space@@YA?AURetVal@0@H@Z"
}


template <class T1, class T2, class T3, class T4, class T5, class T6, class T7,
          class T8, class T9, class T10>
struct Fooob {};

using A0 = Fooob<int, int, int, int, int, int, int, int, int, int>;
using A1 = Fooob<A0, A0, A0, A0, A0, A0, A0, A0, A0, A0>;
using A2 = Fooob<A1, A1, A1, A1, A1, A1, A1, A1, A1, A1>;
using A3 = Fooob<A2, A2, A2, A2, A2, A2, A2, A2, A2, A2>;
using A4 = Fooob<A3, A3, A3, A3, A3, A3, A3, A3, A3, A3>;
using A5 = Fooob<A4, A4, A4, A4, A4, A4, A4, A4, A4, A4>;
using A6 = Fooob<A5, A5, A5, A5, A5, A5, A5, A5, A5, A5>;
using A7 = Fooob<A6, A6, A6, A6, A6, A6, A6, A6, A6, A6>;
using A8 = Fooob<A7, A7, A7, A7, A7, A7, A7, A7, A7, A7>;
using A9 = Fooob<A8, A8, A8, A8, A8, A8, A8, A8, A8, A8>;
using A10 = Fooob<A9, A9, A9, A9, A9, A9, A9, A9, A9, A9>;

// This should take milliseconds, not minutes.
void f(A9 a) {}
// CHECK: "?f@@YAXU?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@U?$Fooob@HHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@@Z"


template <class T1, class T2, class T3, class T4, class T5, class T6, class T7,
          class T8, class T9, class T10, class T11, class T12, class T13,
          class T14, class T15, class T16, class T17, class T18, class T19,
          class T20>
struct Food {};

using B0 = Food<int, int, int, int, int, int, int, int, int, int,  int, int, int, int, int, int, int, int, int, int>;
using B1 = Food<B0, B0, B0, B0, B0, B0, B0, B0, B0, B0,  B0, B0, B0, B0, B0, B0, B0, B0, B0, B0>;
using B2 = Food<B1, B0, B0, B0, B0, B0, B0, B0, B0, B0,  B1, B1, B1, B1, B1, B1, B1, B1, B1, B1>;
using B3 = Food<B2, B1, B0, B0, B0, B0, B0, B0, B0, B0,  B2, B2, B2, B2, B2, B2, B2, B2, B2, B2>;
using B4 = Food<B3, B2, B1, B0, B0, B0, B0, B0, B0, B0,  B3, B3, B3, B3, B3, B3, B3, B3, B3, B3>;
using B5 = Food<B4, B3, B2, B1, B0, B0, B0, B0, B0, B0,  B4, B4, B4, B4, B4, B4, B4, B4, B4, B4>;
using B6 = Food<B5, B4, B3, B2, B1, B0, B0, B0, B0, B0,  B5, B5, B5, B5, B5, B5, B5, B5, B5, B5>;

// This too should take milliseconds, not minutes.
void f(B6 a) {}

// CHECK: "?f@@YAXU?$Food@U?$Food@U?$Food@U?$Food@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U3@U3@U3@U3@U3@U3@U3@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U4@U4@U4@U4@U4@U4@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U3@U3@U3@U3@U3@U3@U3@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U5@U5@U5@U5@U5@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U3@U3@U3@U3@U3@U3@U3@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U4@U4@U4@U4@U4@U4@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U3@U3@U3@U3@U3@U3@U3@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U2@U2@U2@U2@U2@U2@U2@U2@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@U?$Food@HHHHHHHHHHHHHHHHHHHH@@U6@U6@U6@U6@U1@U1@U1@U1@U1@U1@U1@U1@U1@U1@@@@Z"


// Similar to the previous case, except that the later arguments aren't
// present in the earlier ones and hence aren't in the backref cache.
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7,
          class T8, class T9, class T10, class T11, class T12, class T13,
          class T14, class T15, class T16, class T17, class T18, class T19,
          class T20>
struct Fooe {};

using C0 = Fooe<int, int, int, int, int, int, int, int, int, int,  int, int, int, int, int, int, int, int, int, int>;
using C1 = Fooe<C0, C0, C0, C0, C0, C0, C0, C0, C0, C0,  C0, C0, C0, C0, C0, C0, C0, C0, C0, C0>;
using C2 = Fooe<C0, C0, C0, C0, C0, C0, C0, C0, C0, C0,  C1, C1, C1, C1, C1, C1, C1, C1, C1, C1>;
using C3 = Fooe<C1, C1, C0, C0, C0, C0, C0, C0, C0, C0,  C2, C2, C2, C2, C2, C2, C2, C2, C2, C2>;
using C4 = Fooe<C2, C2, C1, C0, C0, C0, C0, C0, C0, C0,  C3, C3, C3, C3, C3, C3, C3, C3, C3, C3>;
using C5 = Fooe<C3, C3, C2, C1, C0, C0, C0, C0, C0, C0,  C4, C4, C4, C4, C4, C4, C4, C4, C4, C4>;
using C6 = Fooe<C4, C4, C3, C2, C1, C0, C0, C0, C0, C0,  C5, C5, C5, C5, C5, C5, C5, C5, C5, C5>;
using C7 = Fooe<C5, C4, C3, C2, C1, C0, C0, C0, C0, C0,  C6, C6, C6, C6, C6, C6, C6, C6, C6, C6>;

// This too should take milliseconds, not minutes.
void f(C7 a) {}
// CHECK: "??@f23afdfb44276eaa53a5575352cf0ebc@"
