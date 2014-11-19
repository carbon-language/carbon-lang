// RUN: %clang_cc1 -verify -fsyntax-only -Wstring-conversion %s

#define assert(EXPR) (void)(EXPR);

// Expection for common assert form.
void test1() {
  assert(0 && "foo");
  assert("foo" && 0);
  assert(0 || "foo"); // expected-warning {{string literal}}
}

void test2() {
  if ("hi") {}           // expected-warning {{string literal}}
  while ("hello") {}     // expected-warning {{string literal}}
  for (;"howdy";) {}     // expected-warning {{string literal}}
  do { } while ("hey");  // expected-warning {{string literal}}
}
