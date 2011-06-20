// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.experimental.CString,deadcode.experimental.UnreachableCode -analyzer-store=region -Wno-null-dereference -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -analyzer-checker=core,unix.experimental.CString,deadcode.experimental.UnreachableCode -analyzer-store=region -Wno-null-dereference -verify %s
// RUN: %clang_cc1 -analyze -DVARIANT -analyzer-checker=core,unix.experimental.CString,deadcode.experimental.UnreachableCode -analyzer-store=region -Wno-null-dereference -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -DVARIANT -analyzer-checker=core,unix.experimental.CString,deadcode.experimental.UnreachableCode -analyzer-store=region -Wno-null-dereference -verify %s

//===----------------------------------------------------------------------===
// Declarations
//===----------------------------------------------------------------------===

// Some functions are so similar to each other that they follow the same code
// path, such as memcpy and __memcpy_chk, or memcmp and bcmp. If VARIANT is
// defined, make sure to use the variants instead to make sure they are still
// checked by the analyzer.

// Some functions are implemented as builtins. These should be #defined as
// BUILTIN(f), which will prepend "__builtin_" if USE_BUILTINS is defined.

// Functions that have variants and are also available as builtins should be
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
  return strlen(0); // expected-warning{{Null pointer argument in call to string length function}}
}

size_t strlen_fn() {
  return strlen((char*)&strlen_fn); // expected-warning{{Argument to string length function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

size_t strlen_nonloc() {
label:
  return strlen((char*)&&label); // expected-warning{{Argument to string length function is the address of the label 'label', which is not a null-terminated string}}
}

void strlen_subregion() {
  struct two_strings { char a[2], b[2]; };
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
  return strnlen(0, 3); // expected-warning{{Null pointer argument in call to string length function}}
}

size_t strnlen_fn() {
  return strnlen((char*)&strlen_fn, 3); // expected-warning{{Argument to string length function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

size_t strnlen_nonloc() {
label:
  return strnlen((char*)&&label, 3); // expected-warning{{Argument to string length function is the address of the label 'label', which is not a null-terminated string}}
}

void strnlen_zero() {
  if (strnlen("abc", 0) != 0)
    (void)*(char*)0; // no-warning
  if (strnlen(NULL, 0) != 0) // no-warning
    (void)*(char*)0; // no-warning
}

size_t strnlen_compound_literal() {
  // This used to crash because we don't model the string lengths of
  // compound literals.
  return strnlen((char[]) { 'a', 'b', 0 }, 1);
}

size_t strnlen_unknown_limit(float f) {
  // This used to crash because we don't model the integer values of floats.
  return strnlen("abc", (int)f);
}

void strnlen_is_not_strlen(char *x) {
  if (strnlen(x, 10) != strlen(x))
    (void)*(char*)0; // expected-warning{{null}}
}

void strnlen_at_limit(char *x) {
  size_t len = strnlen(x, 10);
  if (len > 10)
    (void)*(char*)0; // expected-warning{{never executed}}
  if (len == 10)
    (void)*(char*)0; // expected-warning{{null}}
}

void strnlen_less_than_limit(char *x) {
  size_t len = strnlen(x, 10);
  if (len > 10)
    (void)*(char*)0; // expected-warning{{never executed}}
  if (len < 10)
    (void)*(char*)0; // expected-warning{{null}}
}

void strnlen_at_actual(size_t limit) {
  size_t len = strnlen("abc", limit);
  if (len > 3)
    (void)*(char*)0; // expected-warning{{never executed}}
  if (len == 3)
    (void)*(char*)0; // expected-warning{{null}}
}

void strnlen_less_than_actual(size_t limit) {
  size_t len = strnlen("abc", limit);
  if (len > 3)
    (void)*(char*)0; // expected-warning{{never executed}}
  if (len < 3)
    (void)*(char*)0; // expected-warning{{null}}
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
  strcpy(NULL, x); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strcpy_null_src(char *x) {
  strcpy(x, NULL); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strcpy_fn(char *x) {
  strcpy(x, (char*)&strcpy_fn); // expected-warning{{Argument to string copy function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
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
    strcpy(x, y); // expected-warning{{String copy function overflows destination buffer}}
}

void strcpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strcpy(x, y); // no-warning
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
    stpcpy(x, y); // expected-warning{{String copy function overflows destination buffer}}
}

void stpcpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    stpcpy(x, y); // no-warning
}

//===----------------------------------------------------------------------===
// strcat()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strcat_chk BUILTIN(__strcat_chk)
char *__strcat_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define strcat(a,b) __strcat_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define strcat BUILTIN(strcat)
char *strcat(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */


void strcat_null_dst(char *x) {
  strcat(NULL, x); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strcat_null_src(char *x) {
  strcat(x, NULL); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strcat_fn(char *x) {
  strcat(x, (char*)&strcat_fn); // expected-warning{{Argument to string copy function is the address of the function 'strcat_fn', which is not a null-terminated string}}
}

void strcat_effects(char *y) {
  char x[8] = "123";
  size_t orig_len = strlen(x);
  char a = x[0];

  if (strlen(y) != 4)
    return;

  if (strcat(x, y) != x)
    (void)*(char*)0; // no-warning

  if ((int)strlen(x) != (orig_len + strlen(y)))
    (void)*(char*)0; // no-warning
}

void strcat_overflow_0(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strcat(x, y); // expected-warning{{String copy function overflows destination buffer}}
}

void strcat_overflow_1(char *y) {
  char x[4] = "12";
  if (strlen(y) == 3)
    strcat(x, y); // expected-warning{{String copy function overflows destination buffer}}
}

void strcat_overflow_2(char *y) {
  char x[4] = "12";
  if (strlen(y) == 2)
    strcat(x, y); // expected-warning{{String copy function overflows destination buffer}}
}

void strcat_no_overflow(char *y) {
  char x[5] = "12";
  if (strlen(y) == 2)
    strcat(x, y); // no-warning
}

void strcat_symbolic_dst_length(char *dst) {
	strcat(dst, "1234");
	if (strlen(dst) < 4)
		(void)*(char*)0; // no-warning
}

void strcat_symbolic_src_length(char *src) {
	char dst[8] = "1234";
	strcat(dst, src);
	if (strlen(dst) < 4)
		(void)*(char*)0; // no-warning
}

void strcat_unknown_src_length(char *src, int offset) {
	char dst[8] = "1234";
	strcat(dst, &src[offset]);
	if (strlen(dst) < 4)
		(void)*(char*)0; // no-warning
}

// There is no strcat_unknown_dst_length because if we can't get a symbolic
// length for the "before" strlen, we won't be able to set one for "after".

void strcat_too_big(char *dst, char *src) {
	if (strlen(dst) != (((size_t)0) - 2))
		return;
	if (strlen(src) != 2)
		return;
	strcat(dst, src); // expected-warning{{This expression will create a string whose length is too big to be represented as a size_t}}
}


//===----------------------------------------------------------------------===
// strncpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strncpy_chk BUILTIN(__strncpy_chk)
char *__strncpy_chk(char *restrict s1, const char *restrict s2, size_t n, size_t destlen);

#define strncpy(a,b,n) __strncpy_chk(a,b,n,(size_t)-1)

#else /* VARIANT */

#define strncpy BUILTIN(strncpy)
char *strncpy(char *restrict s1, const char *restrict s2, size_t n);

#endif /* VARIANT */


void strncpy_null_dst(char *x) {
  strncpy(NULL, x, 5); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strncpy_null_src(char *x) {
  strncpy(x, NULL, 5); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strncpy_fn(char *x) {
  strncpy(x, (char*)&strcpy_fn, 5); // expected-warning{{Argument to string copy function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
}

void strncpy_effects(char *x, char *y) {
  char a = x[0];

  if (strncpy(x, y, 5) != x)
    (void)*(char*)0; // no-warning

  if (strlen(x) != strlen(y))
    (void)*(char*)0; // expected-warning{{null}}

  if (a != x[0])
    (void)*(char*)0; // expected-warning{{null}}
}

void strncpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strncpy(x, y, 5); // expected-warning{{Size argument is greater than the length of the destination buffer}}
}

void strncpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, 5); // expected-warning{{Size argument is greater than the length of the destination buffer}}
}

void strncpy_no_overflow2(char *y, int n) {
	if (n <= 4)
		return;

  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, n); // expected-warning{{Size argument is greater than the length of the destination buffer}}
}

void strncpy_truncate(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strncpy(x, y, 3); // no-warning
}

void strncpy_no_truncate(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, 3); // no-warning
}

void strncpy_exactly_matching_buffer(char *y) {
	char x[4];
	strncpy(x, y, 4); // no-warning

	// strncpy does not null-terminate, so we have no idea what the strlen is
	// after this.
	if (strlen(x) > 4)
		(void)*(int*)0; // expected-warning{{null}}
}

void strncpy_exactly_matching_buffer2(char *y) {
	if (strlen(y) >= 4)
		return;

	char x[4];
	strncpy(x, y, 4); // no-warning

	// This time, we know that y fits in x anyway.
	if (strlen(x) > 3)
		(void)*(int*)0; // no-warning
}

//===----------------------------------------------------------------------===
// strncat()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strncat_chk BUILTIN(__strncat_chk)
char *__strncat_chk(char *restrict s1, const char *restrict s2, size_t n, size_t destlen);

#define strncat(a,b,c) __strncat_chk(a,b,c, (size_t)-1)

#else /* VARIANT */

#define strncat BUILTIN(strncat)
char *strncat(char *restrict s1, const char *restrict s2, size_t n);

#endif /* VARIANT */


void strncat_null_dst(char *x) {
  strncat(NULL, x, 4); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strncat_null_src(char *x) {
  strncat(x, NULL, 4); // expected-warning{{Null pointer argument in call to string copy function}}
}

void strncat_fn(char *x) {
  strncat(x, (char*)&strncat_fn, 4); // expected-warning{{Argument to string copy function is the address of the function 'strncat_fn', which is not a null-terminated string}}
}

void strncat_effects(char *y) {
  char x[8] = "123";
  size_t orig_len = strlen(x);
  char a = x[0];

  if (strlen(y) != 4)
    return;

  if (strncat(x, y, strlen(y)) != x)
    (void)*(char*)0; // no-warning

  if (strlen(x) != orig_len + strlen(y))
    (void)*(char*)0; // no-warning
}

void strncat_overflow_0(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strncat(x, y, strlen(y)); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}

void strncat_overflow_1(char *y) {
  char x[4] = "12";
  if (strlen(y) == 3)
    strncat(x, y, strlen(y)); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}

void strncat_overflow_2(char *y) {
  char x[4] = "12";
  if (strlen(y) == 2)
    strncat(x, y, strlen(y)); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}

void strncat_overflow_3(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strncat(x, y, 2); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}
void strncat_no_overflow_1(char *y) {
  char x[5] = "12";
  if (strlen(y) == 2)
    strncat(x, y, strlen(y)); // no-warning
}

void strncat_no_overflow_2(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strncat(x, y, 1); // no-warning
}

void strncat_symbolic_dst_length(char *dst) {
  strncat(dst, "1234", 5);
  if (strlen(dst) < 4)
    (void)*(char*)0; // no-warning
}

void strncat_symbolic_src_length(char *src) {
  char dst[8] = "1234";
  strncat(dst, src, 3);
  if (strlen(dst) < 4)
    (void)*(char*)0; // no-warning

  char dst2[8] = "1234";
  strncat(dst2, src, 4); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}

void strncat_unknown_src_length(char *src, int offset) {
  char dst[8] = "1234";
  strncat(dst, &src[offset], 3);
  if (strlen(dst) < 4)
    (void)*(char*)0; // no-warning

  char dst2[8] = "1234";
  strncat(dst2, &src[offset], 4); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}

// There is no strncat_unknown_dst_length because if we can't get a symbolic
// length for the "before" strlen, we won't be able to set one for "after".

void strncat_symbolic_limit(unsigned limit) {
  char dst[6] = "1234";
  char src[] = "567";
  strncat(dst, src, limit); // no-warning
  if (strlen(dst) < 4)
    (void)*(char*)0; // no-warning
  if (strlen(dst) == 4)
    (void)*(char*)0; // expected-warning{{null}}
}

void strncat_unknown_limit(float limit) {
  char dst[6] = "1234";
  char src[] = "567";
  strncat(dst, src, (size_t)limit); // no-warning
  if (strlen(dst) < 4)
    (void)*(char*)0; // no-warning
  if (strlen(dst) == 4)
    (void)*(char*)0; // expected-warning{{null}}
}

void strncat_too_big(char *dst, char *src) {
  if (strlen(dst) != (((size_t)0) - 2))
    return;
  if (strlen(src) != 2)
    return;
  strncat(dst, src, 2); // expected-warning{{This expression will create a string whose length is too big to be represented as a size_t}}
}

//===----------------------------------------------------------------------===
// strcmp()
//===----------------------------------------------------------------------===

#define strcmp BUILTIN(strcmp)
int strcmp(const char * s1, const char * s2);

void strcmp_constant0() {
  if (strcmp("123", "123") != 0)
    (void)*(char*)0; // no-warning
}

void strcmp_constant_and_var_0() {
  char *x = "123";
  if (strcmp(x, "123") != 0)
    (void)*(char*)0; // no-warning
}

void strcmp_constant_and_var_1() {
  char *x = "123";
    if (strcmp("123", x) != 0)
    (void)*(char*)0; // no-warning
}

void strcmp_0() {
  char *x = "123";
  char *y = "123";
  if (strcmp(x, y) != 0)
    (void)*(char*)0; // no-warning
}

void strcmp_1() {
  char *x = "234";
  char *y = "123";
  if (strcmp(x, y) != 1)
    (void)*(char*)0; // no-warning
}

void strcmp_2() {
  char *x = "123";
  char *y = "234";
  if (strcmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcmp_null_0() {
  char *x = NULL;
  char *y = "123";
  strcmp(x, y); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strcmp_null_1() {
  char *x = "123";
  char *y = NULL;
  strcmp(x, y); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strcmp_diff_length_0() {
  char *x = "12345";
  char *y = "234";
  if (strcmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcmp_diff_length_1() {
  char *x = "123";
  char *y = "23456";
  if (strcmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcmp_diff_length_2() {
  char *x = "12345";
  char *y = "123";
  if (strcmp(x, y) != 1)
    (void)*(char*)0; // no-warning
}

void strcmp_diff_length_3() {
  char *x = "123";
  char *y = "12345";
  if (strcmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcmp_embedded_null () {
	if (strcmp("\0z", "\0y") != 0)
		(void)*(char*)0; // no-warning
}

void strcmp_unknown_arg (char *unknown) {
	if (strcmp(unknown, unknown) != 0)
		(void)*(char*)0; // no-warning
}

//===----------------------------------------------------------------------===
// strncmp()
//===----------------------------------------------------------------------===

#define strncmp BUILTIN(strncmp)
int strncmp(const char *s1, const char *s2, size_t n);

void strncmp_constant0() {
  if (strncmp("123", "123", 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncmp_constant_and_var_0() {
  char *x = "123";
  if (strncmp(x, "123", 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncmp_constant_and_var_1() {
  char *x = "123";
  if (strncmp("123", x, 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncmp_0() {
  char *x = "123";
  char *y = "123";
  if (strncmp(x, y, 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncmp_1() {
  char *x = "234";
  char *y = "123";
  if (strncmp(x, y, 3) != 1)
    (void)*(char*)0; // no-warning
}

void strncmp_2() {
  char *x = "123";
  char *y = "234";
  if (strncmp(x, y, 3) != -1)
    (void)*(char*)0; // no-warning
}

void strncmp_null_0() {
  char *x = NULL;
  char *y = "123";
  strncmp(x, y, 3); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strncmp_null_1() {
  char *x = "123";
  char *y = NULL;
  strncmp(x, y, 3); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strncmp_diff_length_0() {
  char *x = "12345";
  char *y = "234";
  if (strncmp(x, y, 5) != -1)
    (void)*(char*)0; // no-warning
}

void strncmp_diff_length_1() {
  char *x = "123";
  char *y = "23456";
  if (strncmp(x, y, 5) != -1)
    (void)*(char*)0; // no-warning
}

void strncmp_diff_length_2() {
  char *x = "12345";
  char *y = "123";
  if (strncmp(x, y, 5) != 1)
    (void)*(char*)0; // no-warning
}

void strncmp_diff_length_3() {
  char *x = "123";
  char *y = "12345";
  if (strncmp(x, y, 5) != -1)
    (void)*(char*)0; // no-warning
}

void strncmp_diff_length_4() {
  char *x = "123";
  char *y = "12345";
  if (strncmp(x, y, 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncmp_diff_length_5() {
  char *x = "012";
  char *y = "12345";
  if (strncmp(x, y, 3) != -1)
    (void)*(char*)0; // no-warning
}

void strncmp_diff_length_6() {
  char *x = "234";
  char *y = "12345";
  if (strncmp(x, y, 3) != 1)
    (void)*(char*)0; // no-warning
}

void strncmp_embedded_null () {
	if (strncmp("ab\0zz", "ab\0yy", 4) != 0)
		(void)*(char*)0; // no-warning
}

//===----------------------------------------------------------------------===
// strcasecmp()
//===----------------------------------------------------------------------===

#define strcasecmp BUILTIN(strcasecmp)
int strcasecmp(const char *s1, const char *s2);

void strcasecmp_constant0() {
  if (strcasecmp("abc", "Abc") != 0)
    (void)*(char*)0; // no-warning
}

void strcasecmp_constant_and_var_0() {
  char *x = "abc";
  if (strcasecmp(x, "Abc") != 0)
    (void)*(char*)0; // no-warning
}

void strcasecmp_constant_and_var_1() {
  char *x = "abc";
    if (strcasecmp("Abc", x) != 0)
    (void)*(char*)0; // no-warning
}

void strcasecmp_0() {
  char *x = "abc";
  char *y = "Abc";
  if (strcasecmp(x, y) != 0)
    (void)*(char*)0; // no-warning
}

void strcasecmp_1() {
  char *x = "Bcd";
  char *y = "abc";
  if (strcasecmp(x, y) != 1)
    (void)*(char*)0; // no-warning
}

void strcasecmp_2() {
  char *x = "abc";
  char *y = "Bcd";
  if (strcasecmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcasecmp_null_0() {
  char *x = NULL;
  char *y = "123";
  strcasecmp(x, y); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strcasecmp_null_1() {
  char *x = "123";
  char *y = NULL;
  strcasecmp(x, y); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strcasecmp_diff_length_0() {
  char *x = "abcde";
  char *y = "aBd";
  if (strcasecmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcasecmp_diff_length_1() {
  char *x = "abc";
  char *y = "aBdef";
  if (strcasecmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcasecmp_diff_length_2() {
  char *x = "aBcDe";
  char *y = "abc";
  if (strcasecmp(x, y) != 1)
    (void)*(char*)0; // no-warning
}

void strcasecmp_diff_length_3() {
  char *x = "aBc";
  char *y = "abcde";
  if (strcasecmp(x, y) != -1)
    (void)*(char*)0; // no-warning
}

void strcasecmp_embedded_null () {
	if (strcasecmp("ab\0zz", "ab\0yy") != 0)
		(void)*(char*)0; // no-warning
}

//===----------------------------------------------------------------------===
// strncasecmp()
//===----------------------------------------------------------------------===

#define strncasecmp BUILTIN(strncasecmp)
int strncasecmp(const char *s1, const char *s2, size_t n);

void strncasecmp_constant0() {
  if (strncasecmp("abc", "Abc", 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncasecmp_constant_and_var_0() {
  char *x = "abc";
  if (strncasecmp(x, "Abc", 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncasecmp_constant_and_var_1() {
  char *x = "abc";
  if (strncasecmp("Abc", x, 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncasecmp_0() {
  char *x = "abc";
  char *y = "Abc";
  if (strncasecmp(x, y, 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncasecmp_1() {
  char *x = "Bcd";
  char *y = "abc";
  if (strncasecmp(x, y, 3) != 1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_2() {
  char *x = "abc";
  char *y = "Bcd";
  if (strncasecmp(x, y, 3) != -1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_null_0() {
  char *x = NULL;
  char *y = "123";
  strncasecmp(x, y, 3); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strncasecmp_null_1() {
  char *x = "123";
  char *y = NULL;
  strncasecmp(x, y, 3); // expected-warning{{Null pointer argument in call to string comparison function}}
}

void strncasecmp_diff_length_0() {
  char *x = "abcde";
  char *y = "aBd";
  if (strncasecmp(x, y, 5) != -1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_diff_length_1() {
  char *x = "abc";
  char *y = "aBdef";
  if (strncasecmp(x, y, 5) != -1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_diff_length_2() {
  char *x = "aBcDe";
  char *y = "abc";
  if (strncasecmp(x, y, 5) != 1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_diff_length_3() {
  char *x = "aBc";
  char *y = "abcde";
  if (strncasecmp(x, y, 5) != -1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_diff_length_4() {
  char *x = "abcde";
  char *y = "aBc";
  if (strncasecmp(x, y, 3) != 0)
    (void)*(char*)0; // no-warning
}

void strncasecmp_diff_length_5() {
  char *x = "abcde";
  char *y = "aBd";
  if (strncasecmp(x, y, 3) != -1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_diff_length_6() {
  char *x = "aBDe";
  char *y = "abc";
  if (strncasecmp(x, y, 3) != 1)
    (void)*(char*)0; // no-warning
}

void strncasecmp_embedded_null () {
	if (strncasecmp("ab\0zz", "ab\0yy", 4) != 0)
		(void)*(char*)0; // no-warning
}
