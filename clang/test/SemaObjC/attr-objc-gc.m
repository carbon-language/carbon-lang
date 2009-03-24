// RUN: clang-cc -fsyntax-only -verify %s
static id __attribute((objc_gc(weak))) a;
static id __attribute((objc_gc(strong))) b;

static id __attribute((objc_gc())) c; // expected-error{{'objc_gc' attribute requires parameter 1 to be a string}}
static id __attribute((objc_gc(123))) d; // expected-error{{'objc_gc' attribute requires parameter 1 to be a string}}
static id __attribute((objc_gc(foo, 456))) e; // expected-error{{attribute requires 1 argument(s)}}
static id __attribute((objc_gc(hello))) f; // expected-warning{{'objc_gc' attribute argument not supported: 'hello'}}
