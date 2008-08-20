// RUN: clang -fsyntax-only -verify %s

// Define this to get vasprintf on Linux
#define _GNU_SOURCE

#include <stdio.h>
#include <stdarg.h>

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
  vsprintf(buf,s,ap); // no-warning
  vsnprintf(buf,2,s,ap); // no-warning
  vsnprintf(buf,2,global_fmt,ap); // expected-warning {{format string is not a string literal}}
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
  vasprintf(&b,L"bar %d",ap); // expected-warning {{incompatible pointer types}}, expected-warning {{should not be a wide string}}
}

void check_asterisk_precision_width(int x) {
  printf("%*d"); // expected-warning {{'*' specified field width is missing a matching 'int' argument}}
  printf("%.*d"); // expected-warning {{'.*' specified field precision is missing a matching 'int' argument}}
  printf("%*d",12,x); // no-warning
  printf("%*d","foo",x); // expected-warning {{field width should have type 'int', but argument has type 'char *'}}
  printf("%.*d","foo",x); // expected-warning {{field precision should have type 'int', but argument has type 'char *'}}
}
