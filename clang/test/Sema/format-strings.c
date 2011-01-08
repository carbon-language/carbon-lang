// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral %s

#include <stdarg.h>
typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE FILE;
int fprintf(FILE *, const char *restrict, ...);
int printf(const char *restrict, ...); // expected-note{{passing argument to parameter here}}
int snprintf(char *restrict, size_t, const char *restrict, ...);
int sprintf(char *restrict, const char *restrict, ...);
int vasprintf(char **, const char *, va_list);
int asprintf(char **, const char *, ...);
int vfprintf(FILE *, const char *restrict, va_list);
int vprintf(const char *restrict, va_list);
int vsnprintf(char *, size_t, const char *, va_list);
int vsprintf(char *restrict, const char *restrict, va_list); // expected-note{{passing argument to parameter here}}

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
         "%*d", 1, 1); // no-warning
  printf("abc\
def"
         "%*d", 1, 1); // no-warning
         
  // <rdar://problem/6079850>, allow 'unsigned' (instead of 'int') to be used for both
  // the field width and precision.  This deviates from C99, but is reasonably safe
  // and is also accepted by GCC.
  printf("%*d", (unsigned) 1, 1); // no-warning  
}

void check_conditional_literal(const char* s, int i) {
  printf(i == 1 ? "yes" : "no"); // no-warning
  printf(i == 0 ? (i == 1 ? "yes" : "no") : "dont know"); // no-warning
  printf(i == 0 ? (i == 1 ? s : "no") : "dont know"); // expected-warning{{format string is not a string literal}}
  printf("yes" ?: "no %d", 1); // expected-warning{{data argument not used by format string}}
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
  printf("%s%lb%d","unix",10,20); // expected-warning {{invalid conversion specifier 'b'}}
  fprintf(fp,"%%%l"); // expected-warning {{incomplete format specifier}}
  sprintf(buf,"%%%%%ld%d%d", 1, 2, 3); // expected-warning{{conversion specifies type 'long' but the argument has type 'int'}}
  snprintf(buf, 2, "%%%%%ld%;%d", 1, 2, 3); // expected-warning{{conversion specifies type 'long' but the argument has type 'int'}} expected-warning {{invalid conversion specifier ';'}}
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

void test10(int x, float f, int i, long long lli) {
  printf("%s"); // expected-warning{{more '%' conversions than data arguments}}
  printf("%@", 12); // expected-warning{{invalid conversion specifier '@'}}
  printf("\0"); // expected-warning{{format string contains '\0' within the string body}}
  printf("xs\0"); // expected-warning{{format string contains '\0' within the string body}}
  printf("%*d\n"); // expected-warning{{'*' specified field width is missing a matching 'int' argument}}
  printf("%*.*d\n", x); // expected-warning{{'.*' specified field precision is missing a matching 'int' argument}}
  printf("%*d\n", f, x); // expected-warning{{field width should have type 'int', but argument has type 'double'}}
  printf("%*.*d\n", x, f, x); // expected-warning{{field precision should have type 'int', but argument has type 'double'}}
  printf("%**\n"); // expected-warning{{invalid conversion specifier '*'}}
  printf("%n", &i); // expected-warning{{use of '%n' in format string discouraged (potentially insecure)}}
  printf("%d%d\n", x); // expected-warning{{more '%' conversions than data arguments}}
  printf("%d\n", x, x); // expected-warning{{data argument not used by format string}}
  printf("%W%d%Z\n", x, x, x); // expected-warning{{invalid conversion specifier 'W'}} expected-warning{{invalid conversion specifier 'Z'}}
  printf("%"); // expected-warning{{incomplete format specifier}}
  printf("%.d", x); // no-warning
  printf("%.", x);  // expected-warning{{incomplete format specifier}}
  printf("%f", 4); // expected-warning{{conversion specifies type 'double' but the argument has type 'int'}}
  printf("%qd", lli);
  printf("hhX %hhX", (unsigned char)10); // no-warning
  printf("llX %llX", (long long) 10); // no-warning
  // This is fine, because there is an implicit conversion to an int.
  printf("%d", (unsigned char) 10); // no-warning
  printf("%d", (long long) 10); // expected-warning{{conversion specifies type 'int' but the argument has type 'long long'}}
  printf("%Lf\n", (long double) 1.0); // no-warning
  printf("%f\n", (long double) 1.0); // expected-warning{{conversion specifies type 'double' but the argument has type 'long double'}}
  // The man page says that a zero precision is okay.
  printf("%.0Lf", (long double) 1.0); // no-warning
  printf("%c\n", "x"); // expected-warning{{conversion specifies type 'int' but the argument has type 'char *'}}
  printf("%c\n", 1.23); // expected-warning{{conversion specifies type 'int' but the argument has type 'double'}}
  printf("Format %d, is %! %f", 1, 2, 4.4); // expected-warning{{invalid conversion specifier '!'}}
}

typedef unsigned char uint8_t;

void should_understand_small_integers() {
  printf("%hhu", (short) 10); // expected-warning{{conversion specifies type 'unsigned char' but the argument has type 'short'}}
  printf("%hu\n", (unsigned char) 1); // expected-warning{{conversion specifies type 'unsigned short' but the argument has type 'unsigned char'}}
  printf("%hu\n", (uint8_t)1); // expected-warning{{conversion specifies type 'unsigned short' but the argument has type 'uint8_t'}}
}

void test11(void *p, char *s) {
  printf("%p", p); // no-warning
  printf("%p", 123); // expected-warning{{conversion specifies type 'void *' but the argument has type 'int'}}
  printf("%.4p", p); // expected-warning{{precision used with 'p' conversion specifier, resulting in undefined behavior}}
  printf("%+p", p); // expected-warning{{flag '+' results in undefined behavior with 'p' conversion specifier}}
  printf("% p", p); // expected-warning{{flag ' ' results in undefined behavior with 'p' conversion specifier}}
  printf("%0p", p); // expected-warning{{flag '0' results in undefined behavior with 'p' conversion specifier}}
  printf("%s", s); // no-warning
  printf("%+s", p); // expected-warning{{flag '+' results in undefined behavior with 's' conversion specifier}}
  printf("% s", p); // expected-warning{{flag ' ' results in undefined behavior with 's' conversion specifier}}
  printf("%0s", p); // expected-warning{{flag '0' results in undefined behavior with 's' conversion specifier}}
}

void test12(char *b) {
  unsigned char buf[4];
  printf ("%.4s\n", buf); // no-warning
  printf ("%.4s\n", &buf); // expected-warning{{conversion specifies type 'char *' but the argument has type 'unsigned char (*)[4]'}}
  
  // Verify that we are checking asprintf
  asprintf(&b, "%d", "asprintf"); // expected-warning{{conversion specifies type 'int' but the argument has type 'char *'}}
}

typedef struct __aslclient *aslclient;
typedef struct __aslmsg *aslmsg;
int asl_log(aslclient asl, aslmsg msg, int level, const char *format, ...) __attribute__((__format__ (__printf__, 4, 5)));
void test_asl(aslclient asl) {
  // Test case from <rdar://problem/7341605>.
  asl_log(asl, 0, 3, "Error: %m"); // no-warning
  asl_log(asl, 0, 3, "Error: %W"); // expected-warning{{invalid conversion specifier 'W'}}
}

// <rdar://problem/7595366>
typedef enum { A } int_t;
void f0(int_t x) { printf("%d\n", x); }

// Unicode test cases.  These are possibly specific to Mac OS X.  If so, they should
// eventually be moved into a separate test.
typedef __WCHAR_TYPE__ wchar_t;

void test_unicode_conversions(wchar_t *s) {
  printf("%S", s); // no-warning
  printf("%s", s); // expected-warning{{conversion specifies type 'char *' but the argument has type 'wchar_t *'}}
  printf("%C", s[0]); // no-warning
  printf("%c", s[0]);
  // FIXME: This test reports inconsistent results. On Windows, '%C' expects
  // 'unsigned short'.
  // printf("%C", 10);
  // FIXME: we report the expected type as 'int*' instead of 'wchar_t*'
  printf("%S", "hello"); // expected-warning{{but the argument has type 'char *'}}
}

// Mac OS X supports positional arguments in format strings.
// This is an IEEE extension (IEEE Std 1003.1).
// FIXME: This is probably not portable everywhere.
void test_positional_arguments() {
  printf("%0$", (int)2); // expected-warning{{position arguments in format strings start counting at 1 (not 0)}}
  printf("%1$*0$d", (int) 2); // expected-warning{{position arguments in format strings start counting at 1 (not 0)}}
  printf("%1$d", (int) 2); // no-warning
  printf("%1$d", (int) 2, 2); // expected-warning{{data argument not used by format string}}
  printf("%1$d%1$f", (int) 2); // expected-warning{{conversion specifies type 'double' but the argument has type 'int'}}
  printf("%1$2.2d", (int) 2); // no-warning
  printf("%2$*1$.2d", (int) 2, (int) 3); // no-warning
  printf("%2$*8$d", (int) 2, (int) 3); // expected-warning{{specified field width is missing a matching 'int' argument}}
  printf("%%%1$d", (int) 2); // no-warning
  printf("%1$d%%", (int) 2); // no-warning
}

// PR 6697 - Handle format strings where the data argument is not adjacent to the format string
void myprintf_PR_6697(const char *format, int x, ...) __attribute__((__format__(printf,1, 3)));
void test_pr_6697() {
  myprintf_PR_6697("%s\n", 1, "foo"); // no-warning
  myprintf_PR_6697("%s\n", 1, (int)0); // expected-warning{{conversion specifies type 'char *' but the argument has type 'int'}}
  // FIXME: Not everything should clearly support positional arguments,
  // but we need a way to identify those cases.
  myprintf_PR_6697("%1$s\n", 1, "foo"); // no-warning
  myprintf_PR_6697("%2$s\n", 1, "foo"); // expected-warning{{data argument position '2' exceeds the number of data arguments (1)}}
  myprintf_PR_6697("%18$s\n", 1, "foo"); // expected-warning{{data argument position '18' exceeds the number of data arguments (1)}}
  myprintf_PR_6697("%1$s\n", 1, (int) 0); // expected-warning{{conversion specifies type 'char *' but the argument has type 'int'}}
}

void rdar8026030(FILE *fp) {
  fprintf(fp, "\%"); // expected-warning{{incomplete format specifier}}
}

void bug7377_bad_length_mod_usage() {
  // Bad length modifiers
  printf("%hhs", "foo"); // expected-warning{{length modifier 'hh' results in undefined behavior or no effect with 's' conversion specifier}}
  printf("%1$zp", (void *)0); // expected-warning{{length modifier 'z' results in undefined behavior or no effect with 'p' conversion specifier}}
  printf("%ls", L"foo"); // no-warning
  printf("%#.2Lf", (long double)1.234); // no-warning

  // Bad flag usage
  printf("%#p", (void *) 0); // expected-warning{{flag '#' results in undefined behavior with 'p' conversion specifier}}
  printf("%0d", -1); // no-warning
  printf("%#n", (void *) 0); // expected-warning{{flag '#' results in undefined behavior with 'n' conversion specifier}} expected-warning{{use of '%n' in format string discouraged (potentially insecure)}}
  printf("%-n", (void *) 0); // expected-warning{{flag '-' results in undefined behavior with 'n' conversion specifier}} expected-warning{{use of '%n' in format string discouraged (potentially insecure)}}
  printf("%-p", (void *) 0); // no-warning

  // Bad optional amount use
  printf("%.2c", 'a'); // expected-warning{{precision used with 'c' conversion specifier, resulting in undefined behavior}}
  printf("%1n", (void *) 0); // expected-warning{{field width used with 'n' conversion specifier, resulting in undefined behavior}} expected-warning{{use of '%n' in format string discouraged (potentially insecure)}}
  printf("%.9n", (void *) 0); // expected-warning{{precision used with 'n' conversion specifier, resulting in undefined behavior}} expected-warning{{use of '%n' in format string discouraged (potentially insecure)}}

  // Ignored flags
  printf("% +f", 1.23); // expected-warning{{flag ' ' is ignored when flag '+' is present}}
  printf("%+ f", 1.23); // expected-warning{{flag ' ' is ignored when flag '+' is present}}
  printf("%0-f", 1.23); // expected-warning{{flag '0' is ignored when flag '-' is present}}
  printf("%-0f", 1.23); // expected-warning{{flag '0' is ignored when flag '-' is present}}
  printf("%-+f", 1.23); // no-warning
}

// PR 7981 - handle '%lc' (wint_t)
#ifndef wint_t
typedef int __darwin_wint_t;
typedef __darwin_wint_t wint_t;
#endif

void pr7981(wint_t c, wchar_t c2) {
  printf("%lc", c); // no-warning
  printf("%lc", 1.0); // expected-warning{{the argument has type 'double'}}
  printf("%lc", (char) 1); // no-warning
  printf("%lc", &c); // expected-warning{{the argument has type 'wint_t *' (aka 'int *')}}
  printf("%lc", c2); // no-warning
}

// <rdar://problem/8269537> -Wformat-security says NULL is not a string literal
void rdar8269537() {
  // This is likely to crash in most cases, but -Wformat-nonliteral technically
  // doesn't warn in this case.
  printf(0); // no-warning
}

// Handle functions with multiple format attributes.
extern void rdar8332221_vprintf_scanf(const char *, va_list, const char *, ...)
     __attribute__((__format__(__printf__, 1, 0)))
     __attribute__((__format__(__scanf__, 3, 4)));
     
void rdar8332221(va_list ap, int *x, long *y) {
  rdar8332221_vprintf_scanf("%", ap, "%d", x); // expected-warning{{incomplete format specifier}}
}

// PR8641
void pr8641() {
  printf("%#x\n", 10);
  printf("%#X\n", 10);
}

void posix_extensions() {
  // Test %'d, "thousands grouping".
  // <rdar://problem/8816343>
  printf("%'d\n", 123456789); // no-warning
  printf("%'i\n", 123456789); // no-warning
  printf("%'f\n", (float) 1.0); // no-warning
  printf("%'p\n", (void*) 0); // expected-warning{{results in undefined behavior with 'p' conversion specifier}}
}
