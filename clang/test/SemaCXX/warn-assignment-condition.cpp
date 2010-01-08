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
  if (x = 7) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  if ((x = 7)) {}
  do {
  } while (x = 7); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  do {
  } while ((x = 7));
  while (x = 7) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  while ((x = 7)) {}
  for (; x = 7; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  for (; (x = 7); ) {}

  if (p = p) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  if ((p = p)) {}
  do {
  } while (p = p); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  do {
  } while ((p = p));
  while (p = p) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  while ((p = p)) {}
  for (; p = p; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  for (; (p = p); ) {}

  // Initializing variables (shouldn't warn).
  if (int y = x) {}
  while (int y = x) {}
  if (A y = a) {}
  while (A y = a) {}

  // With temporaries.
  if (x = (b+b).foo()) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  if ((x = (b+b).foo())) {}
  do {
  } while (x = (b+b).foo()); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  do {
  } while ((x = (b+b).foo()));
  while (x = (b+b).foo()) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  while ((x = (b+b).foo())) {}
  for (; x = (b+b).foo(); ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  for (; (x = (b+b).foo()); ) {}

  // With a user-defined operator.
  if (a = b + b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  if ((a = b + b)) {}
  do {
  } while (a = b + b); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  do {
  } while ((a = b + b));
  while (a = b + b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  while ((a = b + b)) {}
  for (; a = b + b; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  for (; (a = b + b); ) {}
}
