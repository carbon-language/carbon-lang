// RUN: clang-cc -fsyntax-only -verify %s

// This test creates cases where implicit instantiations of various entities
// would cause a diagnostic, but provides expliict specializations for those
// entities that avoid the diagnostic. The specializations are alternately
// declarations and definitions, and the intent of this test is to verify
// that we allow specializations only in the appropriate namespaces (and
// nowhere else).
struct NonDefaultConstructible {
  NonDefaultConstructible(int);
};


// C++ [temp.expl.spec]p1:
//   An explicit specialization of any of the following:

//     -- function template
namespace N0 {
  template<typename T> void f0(T) { // expected-note{{here}}
    T t;
  }

  template<> void f0(NonDefaultConstructible) { }

  void test_f0(NonDefaultConstructible NDC) {
    f0(NDC);
  }
  
  template<> void f0(int);
  template<> void f0(long);
}

template<> void N0::f0(int) { } // okay

namespace N1 {
  template<> void N0::f0(long) { } // expected-error{{not in a namespace enclosing}}
}

template<> void N0::f0(double) { } // expected-error{{originally be declared}}

struct X1 {
  template<typename T> void f(T);
  
  template<> void f(int); // expected-error{{in class scope}}
};

//     -- class template
namespace N0 {
  
template<typename T>
struct X0 { // expected-note 2{{here}}
  static T member;
  
  void f1(T t) { // expected-note{{explicitly specialized declaration is here}}
    t = 17;
  }
  
  struct Inner : public T { };
  
  template<typename U>
  struct InnerTemplate : public T { };
  
  template<typename U>
  void ft1(T t, U u);
};

}

template<typename T> 
template<typename U>
void N0::X0<T>::ft1(T t, U u) {
  t = u;
}

template<typename T> T N0::X0<T>::member;

template<> struct N0::X0<void> { }; // expected-error{{originally}}
N0::X0<void> test_X0;

namespace N1 {
  template<> struct N0::X0<const void> { }; // expected-error{{originally}}
}

namespace N0 {
  template<> struct X0<volatile void>;
}

template<> struct N0::X0<volatile void> { 
  void f1(void *);
};

//     -- member function of a class template
template<> void N0::X0<void*>::f1(void *) { } // expected-error{{member function specialization}}

void test_spec(N0::X0<void*> xvp, void *vp) {
  xvp.f1(vp);
}

namespace N0 {
  template<> void X0<volatile void>::f1(void *) { } // expected-error{{no function template matches}}
}

#if 0
// FIXME: update the remainder of this test to check for scopes properly.
//     -- static data member of a class template
template<> 
NonDefaultConstructible X0<NonDefaultConstructible>::member = 17;

NonDefaultConstructible &get_static_member() {
  return X0<NonDefaultConstructible>::member;
}

//    -- member class of a class template
template<>
struct X0<void*>::Inner { };

X0<void*>::Inner inner0;

//    -- member class template of a class template
template<>
template<>
struct X0<void*>::InnerTemplate<int> { };

X0<void*>::InnerTemplate<int> inner_template0;

//    -- member function template of a class template
template<>
template<>
void X0<void*>::ft1(void*, const void*) { }

void test_func_template(X0<void *> xvp, void *vp, const void *cvp) {
  xvp.ft1(vp, cvp);
}
#endif
