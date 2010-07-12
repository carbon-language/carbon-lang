// RUN: %clang_cc1 -fsyntax-only -verify %s

int foo(int x) {
  return x == x; // expected-warning {{self-comparison always evaluates to true}}
}

int foo2(int x) {
  return (x) != (((x))); // expected-warning {{self-comparison always evaluates to false}}
}

void foo3(short s, short t) { 
  if (s == s) {} // expected-warning {{self-comparison always evaluates to true}}
  if (s == t) {} // no-warning
}

void foo4(void* v, void* w) {
  if (v == v) {} // expected-warning {{self-comparison always evaluates to true}}
  if (v == w) {} // no-warning
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

#define IS_THE_ANSWER(x) (x == 42)

int macro_comparison() {
  return IS_THE_ANSWER(42);
}

// Don't complain in unevaluated contexts.
int compare_sizeof(int x) {
  return sizeof(x == x); // no-warning
}

int array_comparisons() {
  int array1[2];
  int array2[2];

  //
  // compare same array
  //
  return array1 == array1; // expected-warning{{self-comparison always evaluates to true}}
  return array1 != array1; // expected-warning{{self-comparison always evaluates to false}}
  return array1 < array1; // expected-warning{{self-comparison always evaluates to false}}
  return array1 <= array1; // expected-warning{{self-comparison always evaluates to true}}
  return array1 > array1; // expected-warning{{self-comparison always evaluates to false}}
  return array1 >= array1; // expected-warning{{self-comparison always evaluates to true}}

  //
  // compare differrent arrays
  //
  return array1 == array2; // expected-warning{{array comparison always evaluates to false}}
  return array1 != array2; // expected-warning{{array comparison always evaluates to true}}

  //
  // we don't know what these are going to be
  //
  return array1 < array2; // expected-warning{{array comparison always evaluates to a constant}}
  return array1 <= array2; // expected-warning{{array comparison always evaluates to a constant}}
  return array1 > array2; // expected-warning{{array comparison always evaluates to a constant}}
  return array1 >= array2; // expected-warning{{array comparison always evaluates to a constant}}

}

