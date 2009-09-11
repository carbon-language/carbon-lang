// RUN: clang-cc -fsyntax-only -verify %s

// elaborated-type-specifier:
//   class-key '::'? nested-name-specifier? 'template'? simple-template-id
// Tests that this form is accepted by the compiler but does not follow
// the elaborated lookup rules of [basic.lookup.elab].

template <typename> class Ident {};

namespace A {
  template <typename> void Ident();

  class Ident<int> AIdent; // expected-error {{refers to a function template}}

  // FIXME: this note should be on the template declaration, not the point of instantiation
  class ::Ident<int> AnotherIdent; // expected-note {{previous use is here}}
}

class Ident<int> GlobalIdent;
union Ident<int> GlobalIdent; // expected-error {{ tag type that does not match }}
