// RUN: %clang_cc1 -analyze -analyzer-checker=core.experimental.CString,core.experimental.UnreachableCode -analyzer-check-objc-mem -analyzer-store=region -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -analyzer-checker=core.experimental.CString,core.experimental.UnreachableCode -analyzer-check-objc-mem -analyzer-store=region -verify %s
// RUN: %clang_cc1 -analyze -DVARIANT -analyzer-checker=core.experimental.CString,core.experimental.UnreachableCode -analyzer-check-objc-mem -analyzer-store=region -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -DVARIANT -analyzer-checker=core.experimental.CString,core.experimental.UnreachableCode -analyzer-check-objc-mem -analyzer-store=region -verify %s

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

#define NULL 0
typedef typeof(sizeof(int)) size_t;

//===----------------------------------------------------------------------===
// strlen()
//===----------------------------------------------------------------------===

#define strlen BUILTIN(strlen)
size_t strlen(const char *s);

void strlen_constant0() {
  if (strlen("123") != 3)
    (void)*(char*)0; // no-warning
}

void strlen_constant1() {
  const char *a = "123";
  if (strlen(a) != 3)
    (void)*(char*)0; // no-warning
}

void strlen_constant2(char x) {
  char a[] = "123";
  if (strlen(a) != 3)
    (void)*(char*)0; // no-warning
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

void strlen_subregion() {
  struct two_strings { char a[2], b[2] };
  extern void use_two_strings(struct two_strings *);

  struct two_strings z;
  use_two_strings(&z);

  size_t a = strlen(z.a);
  z.b[0] = 5;
  size_t b = strlen(z.a);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  use_two_strings(&z);

  size_t c = strlen(z.a);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}
}

extern void use_string(char *);
void strlen_argument(char *x) {
  size_t a = strlen(x);
  size_t b = strlen(x);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  use_string(x);

  size_t c = strlen(x);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}  
}

extern char global_str[];
void strlen_global() {
  size_t a = strlen(global_str);
  size_t b = strlen(global_str);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  // Call a function with unknown effects, which should invalidate globals.
  use_string(0);

  size_t c = strlen(global_str);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}  
}

void strlen_indirect(char *x) {
  size_t a = strlen(x);
  char *p = x;
  char **p2 = &p;
  size_t b = strlen(x);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  extern void use_string_ptr(char*const*);
  use_string_ptr(p2);

  size_t c = strlen(x);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}
}

void strlen_liveness(const char *x) {
  if (strlen(x) < 5)
    return;
  if (strlen(x) < 5)
    (void)*(char*)0; // no-warning
}

//===----------------------------------------------------------------------===
// strnlen()
//===----------------------------------------------------------------------===

#define strnlen BUILTIN(strnlen)
size_t strnlen(const char *s, size_t maxlen);

void strnlen_constant0() {
  if (strnlen("123", 10) != 3)
    (void)*(char*)0; // no-warning
}

void strnlen_constant1() {
  const char *a = "123";
  if (strnlen(a, 10) != 3)
    (void)*(char*)0; // no-warning
}

void strnlen_constant2(char x) {
  char a[] = "123";
  if (strnlen(a, 10) != 3)
    (void)*(char*)0; // no-warning
  a[0] = x;
  if (strnlen(a, 10) != 3)
    (void)*(char*)0; // expected-warning{{null}}
}

void strnlen_constant4() {
  if (strnlen("123456", 3) != 3)
    (void)*(char*)0; // no-warning
}

void strnlen_constant5() {
  const char *a = "123456";
  if (strnlen(a, 3) != 3)
    (void)*(char*)0; // no-warning
}

void strnlen_constant6(char x) {
  char a[] = "123456";
  if (strnlen(a, 3) != 3)
    (void)*(char*)0; // no-warning
  a[0] = x;
  if (strnlen(a, 3) != 3)
    (void)*(char*)0; // expected-warning{{null}}
}

size_t strnlen_null() {
  return strnlen(0, 3); // expected-warning{{Null pointer argument in call to byte string function}}
}

size_t strnlen_fn() {
  return strnlen((char*)&strlen_fn, 3); // expected-warning{{Argument to byte string function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

size_t strnlen_nonloc() {
label:
  return strnlen((char*)&&label, 3); // expected-warning{{Argument to byte string function is the address of the label 'label', which is not a null-terminated string}}
}

void strnlen_subregion() {
  struct two_stringsn { char a[2], b[2] };
  extern void use_two_stringsn(struct two_stringsn *);

  struct two_stringsn z;
  use_two_stringsn(&z);

  size_t a = strnlen(z.a, 10);
  z.b[0] = 5;
  size_t b = strnlen(z.a, 10);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  use_two_stringsn(&z);

  size_t c = strnlen(z.a, 10);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}
}

extern void use_stringn(char *);
void strnlen_argument(char *x) {
  size_t a = strnlen(x, 10);
  size_t b = strnlen(x, 10);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  use_stringn(x);

  size_t c = strnlen(x, 10);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}  
}

extern char global_strn[];
void strnlen_global() {
  size_t a = strnlen(global_strn, 10);
  size_t b = strnlen(global_strn, 10);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  // Call a function with unknown effects, which should invalidate globals.
  use_stringn(0);

  size_t c = strnlen(global_str, 10);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}  
}

void strnlen_indirect(char *x) {
  size_t a = strnlen(x, 10);
  char *p = x;
  char **p2 = &p;
  size_t b = strnlen(x, 10);
  if (a == 0 && b != 0)
    (void)*(char*)0; // expected-warning{{never executed}}

  extern void use_stringn_ptr(char*const*);
  use_stringn_ptr(p2);

  size_t c = strnlen(x, 10);
  if (a == 0 && c != 0)
    (void)*(char*)0; // expected-warning{{null}}
}

void strnlen_liveness(const char *x) {
  if (strnlen(x, 10) < 5)
    return;
  if (strnlen(x, 10) < 5)
    (void)*(char*)0; // no-warning
}

//===----------------------------------------------------------------------===
// strcpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strcpy_chk BUILTIN(__strcpy_chk)
char *__strcpy_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define strcpy(a,b) __strcpy_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define strcpy BUILTIN(strcpy)
char *strcpy(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */


void strcpy_null_dst(char *x) {
  strcpy(NULL, x); // expected-warning{{Null pointer argument in call to byte string function}}
}

void strcpy_null_src(char *x) {
  strcpy(x, NULL); // expected-warning{{Null pointer argument in call to byte string function}}
}

void strcpy_fn(char *x) {
  strcpy(x, (char*)&strcpy_fn); // expected-warning{{Argument to byte string function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
}

void strcpy_effects(char *x, char *y) {
  char a = x[0];

  if (strcpy(x, y) != x)
    (void)*(char*)0; // no-warning

  if (strlen(x) != strlen(y))
    (void)*(char*)0; // no-warning

  if (a != x[0])
    (void)*(char*)0; // expected-warning{{null}}
}

void strcpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strcpy(x, y); // expected-warning{{Byte string function overflows destination buffer}}
}

void strcpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strcpy(x, y); // no-warning
}

//===----------------------------------------------------------------------===
// strncpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strncpy_chk BUILTIN(__strncpy_chk)
char *__strncpy_chk(char *restrict s1, const char *restrict s2, size_t n, size_t destlen);

#define strncpy(a,b,c) __strncpy_chk(a,b,c, (size_t)-1)

#else /* VARIANT */

#define strncpy BUILTIN(strncpy)
char *strncpy(char *restrict s1, const char *restrict s2, size_t n);

#endif /* VARIANT */


void strncpy_null_dst(char *x) {
  strncpy(NULL, x, 1); // expected-warning{{Null pointer argument in call to byte string function}}
}

void strncpy_null_src(char *x) {
  strncpy(x, NULL, 1); // expected-warning{{Null pointer argument in call to byte string function}}
}

void strncpy_fn(char *x) {
  strncpy(x, (char*)&strncpy_fn, 1); // expected-warning{{Argument to byte string function is the address of the function 'strncpy_fn', which is not a null-terminated string}}
}

void strncpy_effects(char *x, char *y) {
  char a = x[0];

  if (strncpy(x, y, strlen(y)) != x)
    (void)*(char*)0; // no-warning

  if (strlen(x) != strlen(y))
    (void)*(char*)0; // no-warning

  if (a != x[0])
    (void)*(char*)0; // expected-warning{{null}}
}

void strncpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strncpy(x, y, strlen(y)); // expected-warning{{Byte string function overflows destination buffer}}
}

void strncpy_len_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, sizeof(x)); // no-warning
}

void strncpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, strlen(y)); // no-warning
}

void strncpy_no_len_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strncpy(x, y, sizeof(x)-1); // no-warning
}

//===----------------------------------------------------------------------===
// stpcpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __stpcpy_chk BUILTIN(__stpcpy_chk)
char *__stpcpy_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define stpcpy(a,b) __stpcpy_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define stpcpy BUILTIN(stpcpy)
char *stpcpy(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */


void stpcpy_effect(char *x, char *y) {
  char a = x[0];

  if (stpcpy(x, y) != &x[strlen(y)])
    (void)*(char*)0; // no-warning

  if (strlen(x) != strlen(y))
    (void)*(char*)0; // no-warning

  if (a != x[0])
    (void)*(char*)0; // expected-warning{{null}}
}

void stpcpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    stpcpy(x, y); // expected-warning{{Byte string function overflows destination buffer}}
}

void stpcpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    stpcpy(x, y); // no-warning
}
