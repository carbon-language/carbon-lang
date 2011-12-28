// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
}

void f(char **sp, float *fp) {
  // TODO: Warn that the 'a' length modifier is an extension.
  scanf("%as", sp);

  // TODO: Warn that the 'a' conversion specifier is a C++11 feature.
  printf("%a", 1.0);
  scanf("%afoobar", fp);
}
