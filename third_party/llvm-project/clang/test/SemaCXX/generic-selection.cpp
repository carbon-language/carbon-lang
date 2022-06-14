// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template <typename T, typename U = void*>
struct A {
  enum {
    id = _Generic(T(), // expected-error {{controlling expression type 'char' not compatible with any generic association type}}
        int: 1, // expected-note {{compatible type 'int' specified here}}
        float: 2,
        U: 3) // expected-error {{type 'int' in generic association compatible with previously specified type 'int'}}
  };
};

static_assert(A<int>::id == 1, "fail");
static_assert(A<float>::id == 2, "fail");
static_assert(A<double, double>::id == 3, "fail");

A<char> a1; // expected-note {{in instantiation of template class 'A<char>' requested here}}
A<short, int> a2; // expected-note {{in instantiation of template class 'A<short, int>' requested here}}

template <typename T, typename U>
struct B {
  enum {
    id = _Generic(T(),
        int: 1, // expected-note {{compatible type 'int' specified here}}
        int: 2, // expected-error {{type 'int' in generic association compatible with previously specified type 'int'}}
        U: 3)
  };
};

template <unsigned Arg, unsigned... Args> struct Or {
  enum { result = Arg | Or<Args...>::result };
};

template <unsigned Arg> struct Or<Arg> {
  enum { result = Arg };
};

template <class... Args> struct TypeMask {
  enum {
   result = Or<_Generic(Args(), int: 1, long: 2, short: 4, float: 8)...>::result
  };
};

static_assert(TypeMask<int, long, short>::result == 7, "fail");
static_assert(TypeMask<float, short>::result == 12, "fail");
static_assert(TypeMask<int, float, float>::result == 9, "fail");


struct Test {
  int i;
};

void unreachable_associations(const int i, const Test t) {
  // FIXME: it's not clear to me whether we intended to deviate from the C
  // semantics in terms of how qualifiers are handled, so this documents the
  // existing behavior but perhaps not the desired behavior.
  static_assert(
    _Generic(i,
      const int : 1,    // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'const int' will never be selected because it is qualified}}
      volatile int : 2, // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'volatile int' will never be selected because it is qualified}}
      int[12] : 3,      // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'int[12]' will never be selected because it is of array type}}
      int : 4,
      default : 5
    ) == 4, "we had better pick int, not const int!");
  static_assert(
    _Generic(t,
      Test : 1,
      const Test : 2,  // Ok in C++, warned in C
      default : 3
    ) == 2, "we had better pick const Test, not Test!"); // C++-specific result
}

namespace GH55562 {
struct S { // expected-note {{declared here}}
  int i;
};

void func(struct S s) {
  // We would previously reject this because the parser thought 'struct S :'
  // was the start of a definition (with a base class specifier); it's not, it
  // is an elaborated type specifier followed by the association's value and
  // it should work the same as in C.
  (void)_Generic(s, struct S : 1);

  // The rest of these cases test that we still produce a reasonable diagnostic
  // when referencing an unknown type or trying to define a type in other ways.
  (void)_Generic(s, struct T : 1);            // expected-error {{type 'struct T' in generic association incomplete}}
  (void)_Generic(s, struct U { int a; } : 1); // expected-error {{'U' cannot be defined in a type specifier}}
  (void)_Generic(s, struct V : S);            // expected-error {{'S' does not refer to a value}}
  (void)_Generic(s, struct W : S { int b; } : 1); // expected-error {{expected '(' for function-style cast or type construction}}
}
} // namespace GH55562
