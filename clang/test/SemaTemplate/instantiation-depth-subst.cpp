// RUN: %clang_cc1 -std=c++11 -verify %s -ftemplate-depth 2

// PR9793
template<typename T> auto f(T t) -> decltype(f(t)); // \
// expected-error {{recursive template instantiation exceeded maximum depth of 2}} \
// expected-note 3 {{while substituting}} \
// expected-note {{candidate}}

int k = f(0); // expected-error {{no matching function for call to 'f'}}
