// RUN: %clang_cc1 -fsyntax-only -Wall -Wuninitialized -verify %s

int foo(int x);
int bar(int* x);
int boo(int& x);
int far(const int& x);

// Test self-references within initializers which are guaranteed to be
// uninitialized.
int a = a; // no-warning: used to signal intended lack of initialization.
int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
int c = (c + c); // expected-warning 2 {{variable 'c' is uninitialized when used within its own initialization}}
void test() {
  int d = ({ d + d ;}); // expected-warning {{variable 'd' is uninitialized when used within its own initialization}}
}
int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

// Thes don't warn as they don't require the value.
int g = sizeof(g);
void* ptr = &ptr;
int h = bar(&h);
int i = boo(i);
int j = far(j);
int k = __alignof__(k);

// Also test similar constructs in a field's initializer.
struct S {
  int x;
  void *ptr;

  S(bool (*)[1]) : x(x) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[2]) : x(x + 1) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[3]) : x(x + x) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[4]) : x(static_cast<long>(x) + 1) {} // expected-warning {{field is uninitialized when used here}}
  S(bool (*)[5]) : x(foo(x)) {} // FIXME: This should warn!

  // These don't actually require the value of x and so shouldn't warn.
  S(char (*)[1]) : x(sizeof(x)) {} // rdar://8610363
  S(char (*)[2]) : ptr(&ptr) {}
  S(char (*)[3]) : x(__alignof__(x)) {}
  S(char (*)[4]) : x(bar(&x)) {}
  S(char (*)[5]) : x(boo(x)) {}
  S(char (*)[6]) : x(far(x)) {}
};
