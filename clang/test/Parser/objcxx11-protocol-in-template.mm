// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<class T> class vector {};
@protocol P @end

#if __cplusplus >= 201103L
  // expected-no-diagnostics
#else
  // expected-error@14{{a space is required between consecutive right angle brackets}}
  // expected-error@15{{a space is required between consecutive right angle brackets}}
#endif

vector<id<P>> v;
vector<vector<id<P>>> v2;
