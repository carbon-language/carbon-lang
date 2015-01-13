// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple %ms_abi_triple -DMSABI -fsyntax-only -verify %s

namespace PR5557 {
template <class T> struct A {
  A(); // expected-note{{instantiation}}
  virtual int a(T x);
};
template<class T> A<T>::A() {}

template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

void f() {
  A<int> x; // expected-note{{instantiation}}
}

template<typename T>
struct X {
  virtual void f();
};

template<>
void X<int>::f() { }
}

// Like PR5557, but with a defined destructor instead of a defined constructor.
namespace PR5557_dtor {
template <class T> struct A {
  A(); // Don't have an implicit constructor.
  ~A(); // expected-note{{instantiation}}
  virtual int a(T x);
};
template<class T> A<T>::~A() {}

template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

void f() {
  A<int> x; // expected-note{{instantiation}}
}
}

template<typename T>
struct Base {
  virtual ~Base() { 
    int *ptr = 0;
    T t = ptr; // expected-error{{cannot initialize}}
  }
};

template<typename T>
struct Derived : Base<T> {
  virtual void foo() { }
};

template struct Derived<int>; // expected-note {{in instantiation of member function 'Base<int>::~Base' requested here}}

template<typename T>
struct HasOutOfLineKey {
  HasOutOfLineKey() { } // expected-note{{in instantiation of member function 'HasOutOfLineKey<int>::f' requested here}}
  virtual T *f(float *fp);
};

template<typename T>
T *HasOutOfLineKey<T>::f(float *fp) {
  return fp; // expected-error{{cannot initialize return object of type 'int *' with an lvalue of type 'float *'}}
}

HasOutOfLineKey<int> out_of_line; // expected-note{{in instantiation of member function 'HasOutOfLineKey<int>::HasOutOfLineKey' requested here}}

namespace std {
  class type_info;
}

namespace PR7114 {
  class A { virtual ~A(); }; // expected-note{{declared private here}}

  template<typename T>
  class B {
  public:
    class Inner : public A { }; // expected-error{{base class 'PR7114::A' has private destructor}}
    static Inner i;
    static const unsigned value = sizeof(i) == 4;
  };

  int f() { return B<int>::value; }

#ifdef MSABI
  void test_typeid(B<float>::Inner bfi) { // expected-note{{implicit destructor}}
    (void)typeid(bfi);
#else
  void test_typeid(B<float>::Inner bfi) {
    (void)typeid(bfi); // expected-note{{implicit destructor}}
#endif
  }

  template<typename T>
  struct X : A {
    void f() { }
  };

  void test_X(X<int> &xi, X<float> &xf) {
    xi.f();
  }
}

namespace DynamicCast {
  struct Y {};
  template<typename T> struct X : virtual Y {
    virtual void foo() { T x; } // expected-error {{variable has incomplete type 'void'}}
  };
  template<typename T> struct X2 : virtual Y {
    virtual void foo() { T x; }
  };
  Y* f(X<void>* x) { return dynamic_cast<Y*>(x); } // expected-note {{in instantiation of member function 'DynamicCast::X<void>::foo' requested here}}
  Y* f2(X<void>* x) { return dynamic_cast<Y*>(x); }
}

namespace avoid_using_vtable {
// We shouldn't emit the vtable for this code, in any ABI.  If we emit the
// vtable, we emit an implicit virtual dtor, which calls ~RefPtr, which requires
// a complete type for DeclaredOnly.
//
// Previously we would reference the vtable in the MS C++ ABI, even though we
// don't need to emit either the ctor or the dtor.  In the Itanium C++ ABI, the
// 'trace' method is the key function, so even though we use the vtable, we
// don't emit it.

template <typename T>
struct RefPtr {
  T *m_ptr;
  ~RefPtr() { m_ptr->deref(); }
};
struct DeclaredOnly;
struct Base {
  virtual ~Base();
};

struct AvoidVTable : Base {
  RefPtr<DeclaredOnly> m_insertionStyle;
  virtual void trace();
  AvoidVTable();
};
// Don't call the dtor, because that will emit an implicit dtor, and require a
// complete type for DeclaredOnly.
void foo() { new AvoidVTable; }
}

namespace vtable_uses_incomplete {
// Opposite of the previous test that avoids a vtable, this one tests that we
// use the vtable when the ctor is defined inline.
template <typename T>
struct RefPtr {
  T *m_ptr;
  ~RefPtr() { m_ptr->deref(); }  // expected-error {{member access into incomplete type 'vtable_uses_incomplete::DeclaredOnly'}}
};
struct DeclaredOnly; // expected-note {{forward declaration of 'vtable_uses_incomplete::DeclaredOnly'}}
struct Base {
  virtual ~Base();
};

struct UsesVTable : Base {
  RefPtr<DeclaredOnly> m_insertionStyle;
  virtual void trace();
  UsesVTable() {} // expected-note {{in instantiation of member function 'vtable_uses_incomplete::RefPtr<vtable_uses_incomplete::DeclaredOnly>::~RefPtr' requested here}}
};
}
