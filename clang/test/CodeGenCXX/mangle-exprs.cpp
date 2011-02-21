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
