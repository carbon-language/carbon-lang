// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral %s

typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE FILE;

int fscanf(FILE * restrict, const char * restrict, ...) ;
int scanf(const char * restrict, ...) ;
int sscanf(const char * restrict, const char * restrict, ...) ;

void test(const char *s, int *i) {
  scanf(s, i); // expected-warning{{ormat string is not a string literal}}
  scanf("%0d", i); // expected-warning{{conversion specifies 0 input characters for field width}}
  scanf("%00d", i); // expected-warning{{conversion specifies 0 input characters for field width}}
}
