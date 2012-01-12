/* RUN: %clang_cc1 -fsyntax-only -verify -triple i386-apple-darwin9 -pedantic -std=c89 %s
 */

int scanf(const char * restrict, ...);
int printf(const char *restrict, ...);

void foo(char **sp, float *fp, int *ip) {
  /* TODO: Warn that the 'a' length modifier is an extension. */
  scanf("%as", sp);
  scanf("%a[abc]", sp);

  /* TODO: Warn that the 'a' conversion specifier is a C99 feature. */
  scanf("%a", fp);
  scanf("%afoobar", fp);
  printf("%a", 1.0);
  printf("%as", 1.0);
  printf("%aS", 1.0);
  printf("%a[", 1.0);
  printf("%afoo", 1.0);

  scanf("%da", ip);

  /* Test argument type check for the 'a' length modifier. */
  scanf("%as", fp); /* expected-warning{{conversion specifies type 'char **' but the argument has type 'float *'}} */
  scanf("%aS", fp); /* expected-warning{{conversion specifies type 'wchar_t **' (aka 'int **') but the argument has type 'float *'}} */
  scanf("%a[abc]", fp); /* expected-warning{{conversion specifies type 'char **' but the argument has type 'float *'}} */
}
