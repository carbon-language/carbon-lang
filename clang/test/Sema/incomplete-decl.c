// RUN: clang -fsyntax-only -verify %s

void b;  // expected-error {{variable has incomplete type 'void'}}
struct foo f; // expected-error {{variable has incomplete type 'struct foo'}}

static void c; // expected-error {{variable has incomplete type 'void'}}
static struct foo g;  // expected-error {{variable has incomplete type 'struct foo'}}

extern void d;
extern struct foo e;

void func() {
  void b; // expected-error {{variable has incomplete type 'void'}}
  struct foo f; // expected-error {{variable has incomplete type 'struct foo'}}
}
