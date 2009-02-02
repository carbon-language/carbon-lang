// RUN: clang -fsyntax-only -verify %s
static int f = 10;
static int b = f; // expected-error {{initializer element is not a compile-time constant}}

float r  = (float) &r; // FIXME: should give an error: ptr value used where a float was expected
long long s = (long long) &s;
_Bool t = &t;


union bar {
	int i;
};

struct foo {
	unsigned ptr;
};

union bar u[1];
struct foo x = {(int) u}; // no-error
