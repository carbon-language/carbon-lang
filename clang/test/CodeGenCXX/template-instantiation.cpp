// RUN: %clang_cc1 %s -O1 -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK-NOT: @_ZTVN5test118stdio_sync_filebufIwEE = constant
// CHECK-NOT: _ZTVN5test315basic_fstreamXXIcEE
// CHECK: @_ZTVN5test018stdio_sync_filebufIwEE = unnamed_addr constant

// CHECK-NOT: _ZTVN5test31SIiEE
// CHECK-NOT: _ZTSN5test31SIiEE

// CHECK: define linkonce_odr unnamed_addr void @_ZN5test21CIiEC1Ev(
// CHECK: define linkonce_odr void @_ZN5test21CIiE6foobarIdEEvT_(
// CHECK: define available_externally void @_ZN5test21CIiE6zedbarEd(

namespace test0 {
  struct  basic_streambuf   {
    virtual       ~basic_streambuf();
  };
  template<typename _CharT >
  struct stdio_sync_filebuf : public basic_streambuf {
    virtual void      xsgetn();
  };

  // This specialization should cause the vtable to be emitted, even with
  // the following extern template declaration.
  template<> void stdio_sync_filebuf<wchar_t>::xsgetn()  {
  }
  extern template class stdio_sync_filebuf<wchar_t>;
}

namespace test1 {
  struct  basic_streambuf   {
    virtual       ~basic_streambuf();
  };
  template<typename _CharT >
  struct stdio_sync_filebuf : public basic_streambuf {
    virtual void      xsgetn();
  };

  // Just a declaration should not force the vtable to be emitted.
  template<> void stdio_sync_filebuf<wchar_t>::xsgetn();
}

namespace test2 {
  template<typename T1>
  class C {
  public:
    virtual ~C();
    void zedbar(double) {
    }
    template<typename T2>
    void foobar(T2 foo) {
    }
  };
  extern template class C<int>;
  void g() {
    // The extern template declaration should not prevent us from producing
    // the implicit constructor (test at the top).
    C<int> a;

    // or foobar(test at the top).
    a.foobar(0.0);

    // But it should prevent zebbar
    // (test at the top).
    a.zedbar(0.0);
  }
}

namespace test3 {
  template<typename T>
  class basic_fstreamXX  {
    virtual void foo(){}
    virtual void is_open() const  { }
  };

  extern template class basic_fstreamXX<char>;
  // This template instantiation should not cause us to produce a vtable.
  // (test at the top).
  template void basic_fstreamXX<char>::is_open() const;
}

namespace test3 {
  template <typename T>
  struct S  {
      virtual void m();
  };
  
  template<typename T>
  void S<T>::m() { }

  // Should not cause us to produce vtable because template instantiations
  // don't have key functions.
  template void S<int>::m();
}

namespace test4 {
  template <class T> struct A { static void foo(); };

  class B {
    template <class T> friend void A<T>::foo();
    B();
  };

  template <class T> void A<T>::foo() {
    B b;
  }

  unsigned test() {
    A<int>::foo();
  }
}

namespace PR8505 {
// Hits an assertion due to bogus instantiation of class B.
template <int i> class A {
  class B* g;
};
class B {
  void f () {}
};
// Should not instantiate class B since it is introduced in namespace scope.
// CHECK-NOT: _ZN6PR85051AILi0EE1B1fEv
template class A<0>;
}
