// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -triple=i386-pc-win32 -pedantic %s

int printf(const char *format, ...) __attribute__((format(printf, 1, 2)));

void signed_test() {
  short val = 30;
  printf("val = %I64d\n", val); // expected-warning{{'I64' length modifier is not supported by ISO C}} \
                                // expected-warning{{format specifies type '__int64' (aka 'long long') but the argument has type 'short'}}
  long long bigval = 30;
  printf("val = %I32d\n", bigval); // expected-warning{{'I32' length modifier is not supported by ISO C}} \
                                   // expected-warning{{format specifies type '__int32' (aka 'int') but the argument has type 'long long'}}
  printf("val = %Id\n", bigval); // expected-warning{{'I' length modifier is not supported by ISO C}} \
                                 // expected-warning{{format specifies type '__int32' (aka 'int') but the argument has type 'long long'}}
}

void unsigned_test() {
  unsigned short val = 30;
  printf("val = %I64u\n", val); // expected-warning{{'I64' length modifier is not supported by ISO C}} \
                                // expected-warning{{format specifies type 'unsigned __int64' (aka 'unsigned long long') but the argument has type 'unsigned short'}}
  unsigned long long bigval = 30;
  printf("val = %I32u\n", bigval); // expected-warning{{'I32' length modifier is not supported by ISO C}} \
                                   // expected-warning{{format specifies type 'unsigned __int32' (aka 'unsigned int') but the argument has type 'unsigned long long'}}
  printf("val = %Iu\n", bigval); // expected-warning{{'I' length modifier is not supported by ISO C}} \
                                 // expected-warning{{format specifies type 'unsigned __int32' (aka 'unsigned int') but the argument has type 'unsigned long long'}}
}
