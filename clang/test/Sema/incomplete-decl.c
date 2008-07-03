// RUN: clang -fsyntax-only -verify %s

void b;  // expected-error {{variable has incomplete type 'void'}}
struct foo f; // expected-error {{variable has incomplete type 'struct foo'}}

static void c; // expected-error {{variable has incomplete type 'void'}}
static struct foo g;  // expected-error {{variable has incomplete type 'struct foo'}}

extern void d;
extern struct foo e;

int ary[];
struct foo bary[]; // expected-error {{array has incomplete element type 'struct foo'}}

void func() {
  int ary[]; // expected-error{{variable has incomplete type 'int []'}}
  void b; // expected-error {{variable has incomplete type 'void'}}
  struct foo f; // expected-error {{variable has incomplete type 'struct foo'}}
}

int h[]; 
int (*i)[] = &h+1; // expected-error {{arithmetic on pointer to incomplete type 'int (*)[]'}}

