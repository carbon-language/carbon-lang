// RUN: clang-cc -fsyntax-only -verify %s

namespace N {
  template<class T> class X;
}

// FIXME: this diagnostic is terrible (PR3844).
template<> class X<int> { /* ... */ };	// expected-error {{unqualified-id}}

namespace N {
  
template<> class X<char*> { /* ... */ };	// OK: X is a template
  
}