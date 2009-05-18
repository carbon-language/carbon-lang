// RUN: clang-cc -fsyntax-only -pedantic -verify %s
//
// Tests explicit instantiation of templates.
template<typename T, typename U = T> class X0 { };

namespace N {
  template<typename T, typename U = T> class X1 { };
}

// Check the syntax of explicit instantiations.
template class X0<int, float>;
template class X0<int>; // expected-note{{previous}}

template class N::X1<int>;
template class ::N::X1<int, float>;

using namespace N;
template class X1<float>;

// Check for some bogus syntax that probably means that the user
// wanted to write an explicit specialization, but forgot the '<>'
// after 'template'.
template class X0<double> { }; // expected-error{{explicit specialization}}

// Check for explicit instantiations that come after other kinds of
// instantiations or declarations.
template class X0<int, int>; // expected-error{{duplicate}}

template<> class X0<char> { }; // expected-note{{previous}}
template class X0<char>; // expected-warning{{ignored}}

void foo(X0<short>) { }
template class X0<short>;

// Check that explicit instantiations actually produce definitions. We
// determine whether this happens by placing semantic errors in the
// definition of the template we're instantiating.
template<typename T> struct X2; // expected-note{{declared here}}

template struct X2<float>; // expected-error{{undefined template}}

template<typename T>
struct X2 {
  void f0(T*); // expected-error{{pointer to a reference}}
};

template struct X2<int>; // okay
template struct X2<int&>; // expected-note{{in instantiation of}}

// Check that explicit instantiations instantiate member classes.
template<typename T> struct X3 {
  struct Inner {
    void f(T*); // expected-error{{pointer to a reference}}
  };
};

void f1(X3<int&>); // okay, Inner, not instantiated

template struct X3<int&>; // expected-note{{instantiation}}

template<typename T> struct X4 {
  struct Inner {
    struct VeryInner {
      void f(T*); // expected-error 2{{pointer to a reference}}
    };
  };
};

void f2(X4<int&>); // okay, Inner, not instantiated
void f3(X4<int&>::Inner); // okay, Inner::VeryInner, not instantiated

template struct X4<int&>; // expected-note{{instantiation}}
template struct X4<float&>; // expected-note{{instantiation}}

// Check explicit instantiation of member classes
namespace N2 {

template<typename T>
struct X5 {
  struct Inner1 {
    void f(T&);
  };

  struct Inner2 {
    struct VeryInner {
      void g(T*); // expected-error 2{{pointer to a reference}}
    };
  };
};

}

template struct N2::X5<void>::Inner2;

using namespace N2;
template struct X5<int&>::Inner2; // expected-note{{instantiation}}

void f4(X5<float&>::Inner2);
template struct X5<float&>::Inner2; // expected-note{{instantiation}}

namespace N3 {
  template struct N2::X5<int>::Inner2;
}

struct X6 {
  struct Inner { // expected-note{{here}}
    void f();
  };
};

template struct X6::Inner; // expected-error{{non-templated}}
