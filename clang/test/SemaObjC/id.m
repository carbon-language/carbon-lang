// RUN: clang-cc -fsyntax-only -verify %s

@protocol Foo;

Class T;
id<Foo> S;
id R;
void foo() {
  // Test assignment compatibility of Class and id.  No warning should be
  // produced.
  // rdar://6770142 - Class and id<foo> are compatible.
  S = T; T = S;
  R = T; T = R;
  R = S; S = R;
}

// Test attempt to redefine 'id' in an incompatible fashion.
typedef int id;   // expected-error {{typedef redefinition with different types}}
id b;

