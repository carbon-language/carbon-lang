// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral %s

#include <stdarg.h>
typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE FILE;
typedef __WCHAR_TYPE__ wchar_t;

int fscanf(FILE * restrict, const char * restrict, ...) ;
int scanf(const char * restrict, ...) ;
int sscanf(const char * restrict, const char * restrict, ...) ;
int my_scanf(const char * restrict, ...) __attribute__((__format__(__scanf__, 1, 2)));

int vscanf(const char * restrict, va_list);
int vfscanf(FILE * restrict, const char * restrict, va_list);
int vsscanf(const char * restrict, const char * restrict, va_list);

void test(const char *s, int *i) {
  scanf(s, i); // expected-warning{{ormat string is not a string literal}}
  scanf("%0d", i); // expected-warning{{zero field width in scanf format string is unused}}
  scanf("%00d", i); // expected-warning{{zero field width in scanf format string is unused}}
  scanf("%d%[asdfasdfd", i, s); // expected-warning{{no closing ']' for '%[' in scanf format string}}

  unsigned short s_x;
  scanf ("%" "hu" "\n", &s_x); // no-warning
  scanf("%y", i); // expected-warning{{invalid conversion specifier 'y'}}
  scanf("%%"); // no-warning
  scanf("%%%1$d", i); // no-warning
  scanf("%1$d%%", i); // no-warning
  scanf("%d", i, i); // expected-warning{{data argument not used by format string}}
  scanf("%*d", i); // // expected-warning{{data argument not used by format string}}
  scanf("%*d", i); // // expected-warning{{data argument not used by format string}}
  scanf("%*d%1$d", i); // no-warning
}

void bad_length_modifiers(char *s, void *p, wchar_t *ws, long double *ld) {
  scanf("%hhs", "foo"); // expected-warning{{length modifier 'hh' results in undefined behavior or no effect with 's' conversion specifier}}
  scanf("%1$zp", &p); // expected-warning{{length modifier 'z' results in undefined behavior or no effect with 'p' conversion specifier}}
  scanf("%ls", ws); // no-warning
  scanf("%#.2Lf", ld); // expected-warning{{invalid conversion specifier '#'}}
}

// Test that the scanf call site is where the warning is attached.  If the
// format string is somewhere else, point to it in a note.
void pr9751() {
  int *i;
  char str[100];
  const char kFormat1[] = "%00d"; // expected-note{{format string is defined here}}}
  scanf(kFormat1, i); // expected-warning{{zero field width in scanf format string is unused}}
  scanf("%00d", i); // expected-warning{{zero field width in scanf format string is unused}}
  const char kFormat2[] = "%["; // expected-note{{format string is defined here}}}
  scanf(kFormat2, str); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  scanf("%[", str); // expected-warning{{no closing ']' for '%[' in scanf format string}}
}

void test_variants(int *i, const char *s, ...) {
  FILE *f = 0;
  char buf[100];

  fscanf(f, "%ld", i); // expected-warning{{conversion specifies type 'long *' but the argument has type 'int *'}}
  sscanf(buf, "%ld", i); // expected-warning{{conversion specifies type 'long *' but the argument has type 'int *'}}
  my_scanf("%ld", i); // expected-warning{{conversion specifies type 'long *' but the argument has type 'int *'}}

  va_list ap;
  va_start(ap, s);

  vscanf("%[abc", ap); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  vfscanf(f, "%[abc", ap); // expected-warning{{no closing ']' for '%[' in scanf format string}}
  vsscanf(buf, "%[abc", ap); // expected-warning{{no closing ']' for '%[' in scanf format string}}
}

void test_alloc_extension(char **sp, wchar_t **lsp) {
  /* Make sure "%a" gets parsed as a conversion specifier for float,
   * even when followed by an 's', 'S' or '[', which would cause it to be
   * parsed as a length modifier in C90. */
  scanf("%as", sp); // expected-warning{{conversion specifies type 'float *' but the argument has type 'char **'}}
  scanf("%aS", lsp); // expected-warning{{conversion specifies type 'float *' but the argument has type 'wchar_t **'}}
  scanf("%a[bcd]", sp); // expected-warning{{conversion specifies type 'float *' but the argument has type 'char **'}}
}
