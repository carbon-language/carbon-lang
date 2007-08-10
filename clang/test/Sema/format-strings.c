// RUN: clang -parse-ast-check %s

#include <stdio.h>
#include <stdarg.h>

void check_string_literal( FILE* fp, const char* s, char *buf, ... ) {

  char * b;
  va_list ap;
  va_start(ap,buf);

  printf(s); // expected-warning {{format string is not a string literal}}
  vprintf(s,ap); // expected-warning {{format string is not a string liter}}
  fprintf(fp,s); // expected-warning {{format string is not a string literal}}
  vfprintf(fp,s,ap); // expected-warning {{format string is not a string lit}}
  asprintf(&b,s); // expected-warning {{format string is not a string lit}}
  vasprintf(&b,s,ap); // expected-warning {{format string is not a string lit}}
  sprintf(buf,s); // expected-warning {{format string is not a string literal}}
  snprintf(buf,2,s); // expected-warning {{format string is not a string lit}}
  vsprintf(buf,s,ap); // expected-warning {{format string is not a string lit}}
  vsnprintf(buf,2,s,ap); // expected-warning {{mat string is not a string lit}}
}

