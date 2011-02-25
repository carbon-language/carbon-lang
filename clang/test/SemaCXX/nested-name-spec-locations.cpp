// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note: the formatting in this test case is intentionally funny, with
// nested-name-specifiers stretched out vertically so that we can
// match up diagnostics per-line and still verify that we're getting
// good source-location information.

namespace outer {
  namespace inner {
    template<typename T>
    struct X0 {
    };
  }
}

template<typename T>
struct add_reference {
  typedef T& type;
};

namespace outer_alias = outer;

template<typename T>
struct UnresolvedUsingValueDeclTester {
  using outer::inner::X0<
          typename add_reference<T>::type 
    * // expected-error{{declared as a pointer to a reference of type}}
        >::value;
};

UnresolvedUsingValueDeclTester<int> UnresolvedUsingValueDeclCheck; // expected-note{{in instantiation of template class}}

template<typename T>
struct UnresolvedUsingTypenameDeclTester {
  using outer::inner::X0<
          typename add_reference<T>::type 
    * // expected-error{{declared as a pointer to a reference of type}}
        >::value;
};

UnresolvedUsingTypenameDeclTester<int> UnresolvedUsingTypenameDeclCheck; // expected-note{{in instantiation of template class}}


template<typename T, typename U>
struct PseudoDestructorExprTester {
  void f(T *t) {
    t->T::template Inner<typename add_reference<U>::type 
      * // expected-error{{as a pointer to a reference of type}}
      >::Blarg::~Blarg();
  }
};

struct HasInnerTemplate {
  template<typename T>
  struct Inner;

  typedef HasInnerTemplate T;
};

void PseudoDestructorExprCheck(
                    PseudoDestructorExprTester<HasInnerTemplate, float> tester) {
  tester.f(0); // expected-note{{in instantiation of member function}}
}

template<typename T>
struct DependentScopedDeclRefExpr {
  void f() {
    outer_alias::inner::X0<T>::value = 17;
  }
};
