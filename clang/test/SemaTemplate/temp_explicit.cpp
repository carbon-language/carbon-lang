// RUN: clang-cc -fsyntax-only -verify -pedantic %s
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
