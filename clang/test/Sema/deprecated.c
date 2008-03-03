// RUN: clang %s -verify -fsyntax-only

int f() __attribute__((deprecated));
void g() __attribute__((deprecated));
void g();

void z() __attribute__((bogusattr)); // todo-warning {{'bogusattr' attribute ignored}}

extern int var __attribute__((deprecated));

int a() {
  int (*ptr)() = f; // expected-warning {{'f' is deprecated}}
  f(); // expected-warning {{'f' is deprecated}}

  // test if attributes propagate to functions
  g(); // expected-warning {{'g' is deprecated}}

  return var; // expected-warning {{'var' is deprecated}}
}

// test if attributes propagate to variables
extern int var;
int w() {
	return var; // expected-warning {{'var' is deprecated}}
}
