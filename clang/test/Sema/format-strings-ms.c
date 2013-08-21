// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -triple=i386-pc-win32 -pedantic %s

int printf(const char *format, ...) __attribute__((format(printf, 1, 2)));

void test() {
  short val = 30;
  printf("val = %I64d\n", val); // expected-warning{{'I64' length modifier is not supported by ISO C}} \
                                // expected-warning{{format specifies type '__int64' (aka 'long long') but the argument has type 'short'}}
}
