// RUN: clang -fsyntax-only -Wfloat-equal -verify %s

int foo(float x, float y) {
  return x == y; // expected-warning {{comparing floating point with ==}}
} 

int bar(float x, float y) {
  return x != y; // expected-warning {{comparing floating point with ==}}
}

int qux(float x) {
  return x == x; // no-warning
}

int baz(float x) {
	return x == 0.0; // expected-warning {{comparing}}
}

int taz(float x) {
	return x == __builtin_inf(); // no-warning
}