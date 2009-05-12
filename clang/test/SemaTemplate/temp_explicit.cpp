// RUN: clang-cc -fsyntax-only -verify %s
//
// Tests explicit instantiation of templates.
template<typename T, typename U = T> class X0 { };

namespace N {
  template<typename T, typename U = T> class X1 { };
}

template class X0<int, float>;
template class X0<int>;

template class N::X1<int>;
template class ::N::X1<int, float>;

using namespace N;
template class X1<float>;

template class X0<double> { }; // expected-error{{explicit specialization}}
