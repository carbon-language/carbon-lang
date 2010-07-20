// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral %s

typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE FILE;
typedef __WCHAR_TYPE__ wchar_t;

int fscanf(FILE * restrict, const char * restrict, ...) ;
int scanf(const char * restrict, ...) ;
int sscanf(const char * restrict, const char * restrict, ...) ;

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
  scanf("%1$zp", p); // expected-warning{{length modifier 'z' results in undefined behavior or no effect with 'p' conversion specifier}}
  scanf("%ls", ws); // no-warning
  scanf("%#.2Lf", ld); // expected-warning{{invalid conversion specifier '#'}}
}
