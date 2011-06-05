// RUN: %clang_cc1 -std=c++0x -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

template < bool condition, typename T = void >
struct enable_if { typedef T type; };

template< typename T >
struct enable_if< false, T > {};

// PR5876
namespace Casts {
  template< unsigned O >
  void implicit(typename enable_if< O <= 4 >::type* = 0) {
  }
  
  template< unsigned O >
  void cstyle(typename enable_if< O <= (unsigned)4 >::type* = 0) {
  }

  template< unsigned O >
  void functional(typename enable_if< O <= unsigned(4) >::type* = 0) {
  }
  
  template< unsigned O >
  void static_(typename enable_if< O <= static_cast<unsigned>(4) >::type* = 0) {
  }

  template< typename T >
  void auto_(decltype(new auto(T()))) {
  }

  // FIXME: Test const_cast, reinterpret_cast, dynamic_cast, which are
  // a bit harder to use in template arguments.
  template <unsigned N> struct T {};

  template <int N> T<N> f() { return T<N>(); }
  
  // CHECK: define weak_odr void @_ZN5Casts8implicitILj4EEEvPN9enable_ifIXleT_Li4EEvE4typeE
  template void implicit<4>(void*);
  // CHECK: define weak_odr void @_ZN5Casts6cstyleILj4EEEvPN9enable_ifIXleT_cvjLi4EEvE4typeE
  template void cstyle<4>(void*);
  // CHECK: define weak_odr void @_ZN5Casts10functionalILj4EEEvPN9enable_ifIXleT_cvjLi4EEvE4typeE
  template void functional<4>(void*);
  // CHECK: define weak_odr void @_ZN5Casts7static_ILj4EEEvPN9enable_ifIXleT_cvjLi4EEvE4typeE
  template void static_<4>(void*);

  // CHECK: define weak_odr void @_ZN5Casts1fILi6EEENS_1TIXT_EEEv
  template T<6> f<6>();

  // CHECK: define weak_odr void @_ZN5Casts5auto_IiEEvDTnw_DapicvT__EEE(
  template void auto_<int>(int*);
}

namespace test1 {
  short foo(short);
  int foo(int);

  // CHECK: define linkonce_odr signext i16 @_ZN5test11aIsEEDTcl3foocvT__EEES1_(
  template <class T> auto a(T t) -> decltype(foo(T())) { return foo(t); }

  // CHECK: define linkonce_odr signext i16 @_ZN5test11bIsEEDTcp3foocvT__EEES1_(
  template <class T> auto b(T t) -> decltype((foo)(T())) { return (foo)(t); }

  void test(short s) {
    a(s);
    b(s);
  }
}

namespace test2 {
  template <class T> void a(T x, decltype(x()) y) {}
  template <class T> auto b(T x) -> decltype(x()) { return x(); }
  template <class T> void c(T x, void (*p)(decltype(x()))) {}
  template <class T> void d(T x, auto (*p)() -> decltype(x())) {}
  template <class T> void e(auto (*p)(T y) -> decltype(y())) {}
  template <class T> void f(void (*p)(T x, decltype(x()) y)) {}
  template <class T> void g(T x, decltype(x()) y) {
    static decltype(x()) variable;
    variable = 0;
  }
  template <class T> void h(T x, decltype((decltype(x())(*)()) 0) y) {}
  template <class T> void i(decltype((auto (*)(T x) -> decltype(x())) 0) y) {}

  float foo();
  void bar(float);
  float baz(float(*)());
  void fred(float(*)(), float);

  // CHECK: define void @_ZN5test211instantiateEv
  void instantiate() {
    // CHECK: call void @_ZN5test21aIPFfvEEEvT_DTclfL0p_EE(
    a(foo, 0.0f);
    // CHECK: call float @_ZN5test21bIPFfvEEEDTclfp_EET_(
    (void) b(foo);
    // CHECK: call void @_ZN5test21cIPFfvEEEvT_PFvDTclfL1p_EEE(
    c(foo, bar);
    // CHECK: call void @_ZN5test21dIPFfvEEEvT_PFDTclfL0p_EEvE(
    d(foo, foo);
    // CHECK: call void @_ZN5test21eIPFfvEEEvPFDTclfp_EET_E(
    e(baz);
    // CHECK: call void @_ZN5test21fIPFfvEEEvPFvT_DTclfL0p_EEE(
    f(fred);
    // CHECK: call void @_ZN5test21gIPFfvEEEvT_DTclfL0p_EE(
    g(foo, 0.0f);
    // CHECK: call void @_ZN5test21hIPFfvEEEvT_DTcvPFDTclfL0p_EEvELi0EE(
    h(foo, foo);
    // CHECK: call void @_ZN5test21iIPFfvEEEvDTcvPFDTclfp_EET_ELi0EE(
    i<float(*)()>(baz);
  }

  // CHECK: store float {{.*}}, float* @_ZZN5test21gIPFfvEEEvT_DTclfL0p_EEE8variable,
}

namespace test3 {
  template <class T, class U> void a(T x, U y, decltype(x.*y) z) {}  

  struct X {
    int *member;
  };

  // CHECK: define void @_ZN5test311instantiateEv
  void instantiate() {
    X x;
    int *ip;
    // CHECK: call void @_ZN5test31aINS_1XEMS1_PiEEvT_T0_DTdsfL0p_fL0p0_E
    a(x, &X::member, ip);
  }
}
