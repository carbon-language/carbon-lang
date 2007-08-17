#define CFSTR __builtin___CFStringMakeConstantString

// RUN: clang %s -parse-ast-check
void f() {
  CFSTR("\242"); // expected-warning { CFString literal contains non-ASCII character }
  CFSTR("\0"); // expected-warning { CFString literal contains NUL character }
  CFSTR(242); // expected-error { error: CFString literal is not a string constant } \
  expected-warning { incompatible types }
  CFSTR("foo", "bar"); // expected-error { error: too many arguments to function }
}
