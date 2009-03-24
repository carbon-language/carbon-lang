// RUN: clang-cc -fsyntax-only -verify %s

int foo(int x) {
  return x == x; // expected-warning {{self-comparison always results}}
}

int foo2(int x) {
  return (x) != (((x))); // expected-warning {{self-comparison always results}}
}

int qux(int x) {
   return x < x; // expected-warning {{self-comparison}}
}

int qux2(int x) {
   return x > x; // expected-warning {{self-comparison}}
}

int bar(float x) {
  return x == x; // no-warning
}

int bar2(float x) {
  return x != x; // no-warning
}

// Motivated by <rdar://problem/6703892>, self-comparisons of enum constants
// should not be warned about.  These can be expanded from macros, and thus
// are usually deliberate.
int compare_enum() {
  enum { A };
  return A == A; // no-warning
}
