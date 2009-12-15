// RUN: %clang_cc1 -fsyntax-only -verify %s

struct NonDefaultConstructible {
  NonDefaultConstructible(const NonDefaultConstructible&);
};

template<typename T, typename U>
struct X {
  static T member;
};

template<typename T, typename U>
T X<T, U>::member; // expected-error{{no matching constructor}}

// Okay; this is a declaration, not a definition.
template<>
NonDefaultConstructible X<NonDefaultConstructible, long>::member;

NonDefaultConstructible &test(bool b) {
  return b? X<NonDefaultConstructible, int>::member // expected-note{{instantiation}}
          : X<NonDefaultConstructible, long>::member;
}
