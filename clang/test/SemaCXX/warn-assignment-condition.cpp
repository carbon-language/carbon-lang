// RUN: %clang_cc1 -fsyntax-only -Wparentheses -verify %s

struct A {
  int foo();
  friend A operator+(const A&, const A&);
  operator bool();
};

void test() {
  int x, *p;
  A a, b;

  // With scalars.
  if (x = 7) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  if ((x = 7)) {}
  do {
  } while (x = 7); // expected-warning {{using the result of an assignment as a condition without parentheses}}
  do {
  } while ((x = 7));
  while (x = 7) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  while ((x = 7)) {}
  for (; x = 7; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  for (; (x = 7); ) {}

  if (p = p) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  if ((p = p)) {}
  do {
  } while (p = p); // expected-warning {{using the result of an assignment as a condition without parentheses}}
  do {
  } while ((p = p));
  while (p = p) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  while ((p = p)) {}
  for (; p = p; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  for (; (p = p); ) {}

  // Initializing variables (shouldn't warn).
  if (int y = x) {}
  while (int y = x) {}
  if (A y = a) {}
  while (A y = a) {}

  // With temporaries.
  if (x = (b+b).foo()) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  if ((x = (b+b).foo())) {}
  do {
  } while (x = (b+b).foo()); // expected-warning {{using the result of an assignment as a condition without parentheses}}
  do {
  } while ((x = (b+b).foo()));
  while (x = (b+b).foo()) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  while ((x = (b+b).foo())) {}
  for (; x = (b+b).foo(); ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  for (; (x = (b+b).foo()); ) {}

  // With a user-defined operator.
  if (a = b + b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  if ((a = b + b)) {}
  do {
  } while (a = b + b); // expected-warning {{using the result of an assignment as a condition without parentheses}}
  do {
  } while ((a = b + b));
  while (a = b + b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  while ((a = b + b)) {}
  for (; a = b + b; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}}
  for (; (a = b + b); ) {}
}
