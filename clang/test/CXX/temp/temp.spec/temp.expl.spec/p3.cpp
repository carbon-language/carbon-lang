// RUN: clang-cc -fsyntax-only -verify %s

namespace N {
  template<class T> class X;
}

template<> class X<int> { /* ... */ };	// expected-error {{non-template class 'X'}}

namespace N {
  
template<> class X<char*> { /* ... */ };	// OK: X is a template
  
}
