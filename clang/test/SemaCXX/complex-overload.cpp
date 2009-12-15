// RUN: %clang_cc1 -fsyntax-only -verify %s
char *foo(float); // expected-note 3 {{candidate function}}

void test_foo_1(float fv, double dv, float _Complex fc, double _Complex dc) {
  char *cp1 = foo(fv);
  char *cp2 = foo(dv);
  // Note: GCC and EDG reject these two, but they are valid C99 conversions
  char *cp3 = foo(fc);
  char *cp4 = foo(dc);
}

int *foo(float _Complex); // expected-note 3 {{candidate function}}

void test_foo_2(float fv, double dv, float _Complex fc, double _Complex dc) {
  char *cp1 = foo(fv);
  char *cp2 = foo(dv); // expected-error{{call to 'foo' is ambiguous; candidates are:}}
  int *ip = foo(fc);
  int *lp = foo(dc); // expected-error{{call to 'foo' is ambiguous; candidates are:}}
}

long *foo(double _Complex); // expected-note {{candidate function}}

void test_foo_3(float fv, double dv, float _Complex fc, double _Complex dc) {
  char *cp1 = foo(fv);
  char *cp2 = foo(dv); // expected-error{{call to 'foo' is ambiguous; candidates are:}}
  int *ip = foo(fc);
  long *lp = foo(dc);
}

char *promote_or_convert(double _Complex);  // expected-note{{candidate function}}
int *promote_or_convert(long double _Complex); // expected-note{{candidate function}} 

void test_promote_or_convert(float f, float _Complex fc) {
  char *cp = promote_or_convert(fc);
  int *ip2 = promote_or_convert(f); // expected-error{{call to 'promote_or_convert' is ambiguous; candidates are:}}
}

char *promote_or_convert2(float);
int *promote_or_convert2(double _Complex);

void test_promote_or_convert2(float _Complex fc) {
  int *cp = promote_or_convert2(fc);
}

char *promote_or_convert3(int _Complex);
int *promote_or_convert3(long _Complex);

void test_promote_or_convert3(short _Complex sc) {
  char *cp = promote_or_convert3(sc);
}
