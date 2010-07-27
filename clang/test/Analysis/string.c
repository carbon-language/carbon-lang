// RUN: %clang_cc1 -analyze -Wwrite-strings -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -Wwrite-strings -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DVARIANT -Wwrite-strings -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -DVARIANT -Wwrite-strings -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s

//===----------------------------------------------------------------------===
// Declarations
//===----------------------------------------------------------------------===

// Some functions are so similar to each other that they follow the same code
// path, such as memcpy and __memcpy_chk, or memcmp and bcmp. If VARIANT is
// defined, make sure to use the variants instead to make sure they are still
// checked by the analyzer.

// Some functions are implemented as builtins. These should be #defined as
// BUILTIN(f), which will prepend "__builtin_" if USE_BUILTINS is defined.

// Functions that have variants and are also availabe as builtins should be
// declared carefully! See memcpy() for an example.

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else /* USE_BUILTINS */
# define BUILTIN(f) f
#endif /* USE_BUILTINS */

typedef typeof(sizeof(int)) size_t;

//===----------------------------------------------------------------------===
// strlen()
//===----------------------------------------------------------------------===

#define strlen BUILTIN(strlen)
size_t strlen(const char *s);

void strlen_constant0() {
  if (strlen("123") != 3)
    (void)*(char*)0;
}

void strlen_constant1() {
  const char *a = "123";
  if (strlen(a) != 3)
    (void)*(char*)0;
}

void strlen_constant2(char x) {
  char a[] = "123";
  a[0] = x;
  if (strlen(a) != 3)
    (void)*(char*)0; // expected-warning{{null}}
}

size_t strlen_null() {
  return strlen(0); // expected-warning{{Null pointer argument in call to byte string function}}
}

size_t strlen_fn() {
  return strlen((char*)&strlen_fn); // expected-warning{{Argument to byte string function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

size_t strlen_nonloc() {
label:
  return strlen((char*)&&label); // expected-warning{{Argument to byte string function is the address of the label 'label', which is not a null-terminated string}}
}
