// RUN: %clang_cc1 -fsyntax-only -Wglobal-constructors %s -verify

int opaque_int();

namespace test0 {
  // These should never require global constructors.
  int a;
  int b = 20;
  float c = 5.0f;

  // This global constructor is avoidable based on initialization order.
  int d = b; // expected-warning {{global constructor}}

  // These global constructors are unavoidable.
  int e = opaque_int(); // expected-warning {{global constructor}}
  int f = b; // expected-warning {{global constructor}}
}

namespace test1 {
  struct A { int x; };
  A a;
  A b = A();
  A c = { 10 };
  A d = { opaque_int() }; // expected-warning {{global constructor}}
}

namespace test2 {
  struct A { A(); };
  A a; // expected-warning {{global constructor}}
  A b[10]; // expected-warning {{global constructor}}
  A c[10][10]; // expected-warning {{global constructor}}

  // FIXME: false positives!
  A &d = a; // expected-warning {{global constructor}}
  A &e = b[5]; // expected-warning {{global constructor}}
  A &f = c[5][7]; // expected-warning {{global constructor}}
}

namespace test3 {
  struct A { ~A(); };
  A a; // expected-warning {{global destructor}}
  A b[10]; // expected-warning {{global destructor}}
  A c[10][10]; // expected-warning {{global destructor}}

  // FIXME: false positives!
  A &d = a; // expected-warning {{global constructor}}
  A &e = b[5]; // expected-warning {{global constructor}}
  A &f = c[5][7]; // expected-warning {{global constructor}}
}

namespace test4 {
  char a[] = "hello";
  char b[5] = "hello";
  char c[][5] = { "hello" };
}
