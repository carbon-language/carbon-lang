// RUN: clang-cc -fsyntax-only -verify -Wformat-nonliteral %s

#include <stdarg.h>
typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE FILE;
int fprintf(FILE *, const char *restrict, ...);
int printf(const char *restrict, ...);
int snprintf(char *restrict, size_t, const char *restrict, ...);
int sprintf(char *restrict, const char *restrict, ...);
int vasprintf(char **, const char *, va_list);
int vfprintf(FILE *, const char *restrict, va_list);
int vprintf(const char *restrict, va_list);
int vsnprintf(char *, size_t, const char *, va_list);
int vsprintf(char *restrict, const char *restrict, va_list);

char * global_fmt;

void check_string_literal( FILE* fp, const char* s, char *buf, ... ) {

  char * b;
  va_list ap;
  va_start(ap,buf);

  printf(s); // expected-warning {{format string is not a string literal}}
  vprintf(s,ap); // // no-warning
  fprintf(fp,s); // expected-warning {{format string is not a string literal}}
  vfprintf(fp,s,ap); // no-warning
  asprintf(&b,s); // expected-warning {{format string is not a string lit}}
  vasprintf(&b,s,ap); // no-warning
  sprintf(buf,s); // expected-warning {{format string is not a string literal}}
  snprintf(buf,2,s); // expected-warning {{format string is not a string lit}}
  __builtin___sprintf_chk(buf,0,-1,s); // expected-warning {{format string is not a string literal}}
  __builtin___snprintf_chk(buf,2,0,-1,s); // expected-warning {{format string is not a string lit}}
  vsprintf(buf,s,ap); // no-warning
  vsnprintf(buf,2,s,ap); // no-warning
  vsnprintf(buf,2,global_fmt,ap); // expected-warning {{format string is not a string literal}}
  __builtin___vsnprintf_chk(buf,2,0,-1,s,ap); // no-warning
  __builtin___vsnprintf_chk(buf,2,0,-1,global_fmt,ap); // expected-warning {{format string is not a string literal}}

  // rdar://6079877
  printf("abc"
         "%*d", (unsigned) 1, 1); // expected-warning {{field width should have type 'int'}}
  printf("abc\
def"
         "%*d", (unsigned) 1, 1); // expected-warning {{field width should have type 'int'}}
  
}

void check_conditional_literal(const char* s, int i) {
  printf(i == 1 ? "yes" : "no"); // no-warning
  printf(i == 0 ? (i == 1 ? "yes" : "no") : "dont know"); // no-warning
  printf(i == 0 ? (i == 1 ? s : "no") : "dont know"); // expected-warning{{format string is not a string literal}}
}

void check_writeback_specifier()
{
  int x;
  char *b;

  printf("%n",&x); // expected-warning {{'%n' in format string discouraged}}
  sprintf(b,"%d%%%n",1, &x); // expected-warning {{'%n' in format string dis}}
}

void check_invalid_specifier(FILE* fp, char *buf)
{
  printf("%s%lb%d","unix",10,20); // expected-warning {{lid conversion '%lb'}}
  fprintf(fp,"%%%l"); // expected-warning {{lid conversion '%l'}}
  sprintf(buf,"%%%%%ld%d%d", 1, 2, 3); // no-warning
  snprintf(buf, 2, "%%%%%ld%;%d", 1, 2, 3); // expected-warning {{sion '%;'}}
}

void check_null_char_string(char* b)
{
  printf("\0this is bogus%d",1); // expected-warning {{string contains '\0'}}
  snprintf(b,10,"%%%%%d\0%d",1,2); // expected-warning {{string contains '\0'}}
  printf("%\0d",1); // expected-warning {{string contains '\0'}}
}

void check_empty_format_string(char* buf, ...)
{
  va_list ap;
  va_start(ap,buf);
  vprintf("",ap); // expected-warning {{format string is empty}}
  sprintf(buf,""); // expected-warning {{format string is empty}}
}

void check_wide_string(char* b, ...)
{
  va_list ap;
  va_start(ap,b);

  printf(L"foo %d",2); // expected-warning {{incompatible pointer types}}, expected-warning {{should not be a wide string}}
  vsprintf(b,L"bar %d",ap); // expected-warning {{incompatible pointer types}}, expected-warning {{should not be a wide string}}
}

void check_asterisk_precision_width(int x) {
  printf("%*d"); // expected-warning {{'*' specified field width is missing a matching 'int' argument}}
  printf("%.*d"); // expected-warning {{'.*' specified field precision is missing a matching 'int' argument}}
  printf("%*d",12,x); // no-warning
  printf("%*d","foo",x); // expected-warning {{field width should have type 'int', but argument has type 'char *'}}
  printf("%.*d","foo",x); // expected-warning {{field precision should have type 'int', but argument has type 'char *'}}
}

void __attribute__((format(printf,1,3))) myprintf(const char*, int blah, ...);

void test_myprintf() {
  myprintf("%d", 17, 18); // okay
}

void test_constant_bindings(void) {
  const char * const s1 = "hello";
  const char s2[] = "hello";
  const char *s3 = "hello";
  char * const s4 = "hello";
  extern const char s5[];
  
  printf(s1); // no-warning
  printf(s2); // no-warning
  printf(s3); // expected-warning{{not a string literal}}
  printf(s4); // expected-warning{{not a string literal}}
  printf(s5); // expected-warning{{not a string literal}}
}


// Test what happens when -Wformat-security only.
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic warning "-Wformat-security"

void test9(char *P) {
  int x;
  printf(P);   // expected-warning {{format string is not a string literal (potentially insecure)}}
  printf(P, 42);
  printf("%n", &x); // expected-warning {{use of '%n' in format string discouraged }}
}

void torture(va_list v8) {
  vprintf ("%*.*d", v8);  // no-warning
}

