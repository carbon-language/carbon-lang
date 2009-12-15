// RUN: %clang_cc1 -fsyntax-only -verify %s 

// PR5521
struct A { void a(const char*,...) __attribute((format(printf,2,3))); };
void b(A x) {
  x.a("%d", 3);
}
struct X { void a(const char*,...) __attribute((format(printf,1,3))); }; // expected-error {{format argument not a string type}}
