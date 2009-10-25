// RUN: clang-cc %s -verify -fsyntax-only

int f() __attribute__((deprecated));
void g() __attribute__((deprecated));
void g();

void z() __attribute__((bogusattr)); // expected-warning {{'bogusattr' attribute ignored}}

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

int old_fn() __attribute__ ((deprecated));
int old_fn();
int (*fn_ptr)() = old_fn; // expected-warning {{'old_fn' is deprecated}}

int old_fn() {
  return old_fn()+1;  // no warning, deprecated functions can use deprecated symbols.
}


struct foo {
  int x __attribute__((deprecated));
};

void test1(struct foo *F) {
  ++F->x;  // expected-warning {{'x' is deprecated}}
}

typedef struct foo foo_dep __attribute__((deprecated));
foo_dep *test2;    // expected-warning {{'foo_dep' is deprecated}}

struct bar_dep __attribute__((deprecated, 
                              invalid_attribute));  // expected-warning {{'invalid_attribute' attribute ignored}}

struct bar_dep *test3;   // expected-warning {{'bar_dep' is deprecated}}

