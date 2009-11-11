// RUN: clang-cc -fsyntax-only -verify -Wconversion -triple x86_64-apple-darwin %s

#define BIG 0x7f7f7f7f7f7f7f7fL

void test0(char c, short s, int i, long l, long long ll) {
  c = c;
  c = s; // expected-warning {{implicit cast loses integer precision}}
  c = i; // expected-warning {{implicit cast loses integer precision}}
  c = l; // expected-warning {{implicit cast loses integer precision}}
  s = c;
  s = s;
  s = i; // expected-warning {{implicit cast loses integer precision}}
  s = l; // expected-warning {{implicit cast loses integer precision}}
  i = c;
  i = s;
  i = i;
  i = l; // expected-warning {{implicit cast loses integer precision}}
  l = c;
  l = s;
  l = i;
  l = l;

  c = (char) 0;
  c = (short) 0;
  c = (int) 0;
  c = (long) 0;
  s = (char) 0;
  s = (short) 0;
  s = (int) 0;
  s = (long) 0;
  i = (char) 0;
  i = (short) 0;
  i = (int) 0;
  i = (long) 0;
  l = (char) 0;
  l = (short) 0;
  l = (int) 0;
  l = (long) 0;

  c = (char) BIG;
  c = (short) BIG; // expected-warning {{implicit cast loses integer precision}}
  c = (int) BIG; // expected-warning {{implicit cast loses integer precision}}
  c = (long) BIG; // expected-warning {{implicit cast loses integer precision}}
  s = (char) BIG;
  s = (short) BIG;
  s = (int) BIG; // expected-warning {{implicit cast loses integer precision}}
  s = (long) BIG; // expected-warning {{implicit cast loses integer precision}}
  i = (char) BIG;
  i = (short) BIG;
  i = (int) BIG;
  i = (long) BIG; // expected-warning {{implicit cast loses integer precision}}
  l = (char) BIG;
  l = (short) BIG;
  l = (int) BIG;
  l = (long) BIG;
}

char test1(long long ll) {
  return (long long) ll; // expected-warning {{implicit cast loses integer precision}}
  return (long) ll; // expected-warning {{implicit cast loses integer precision}}
  return (int) ll; // expected-warning {{implicit cast loses integer precision}}
  return (short) ll; // expected-warning {{implicit cast loses integer precision}}
  return (char) ll;
  return (long long) BIG; // expected-warning {{implicit cast loses integer precision}}
  return (long) BIG; // expected-warning {{implicit cast loses integer precision}}
  return (int) BIG; // expected-warning {{implicit cast loses integer precision}}
  return (short) BIG; // expected-warning {{implicit cast loses integer precision}}
  return (char) BIG;
}

short test2(long long ll) {
  return (long long) ll; // expected-warning {{implicit cast loses integer precision}}
  return (long) ll; // expected-warning {{implicit cast loses integer precision}}
  return (int) ll; // expected-warning {{implicit cast loses integer precision}}
  return (short) ll;
  return (char) ll;
  return (long long) BIG;  // expected-warning {{implicit cast loses integer precision}}
  return (long) BIG;  // expected-warning {{implicit cast loses integer precision}}
  return (int) BIG;  // expected-warning {{implicit cast loses integer precision}}
  return (short) BIG;
  return (char) BIG;
}

int test3(long long ll) {
  return (long long) ll;  // expected-warning {{implicit cast loses integer precision}}
  return (long) ll;  // expected-warning {{implicit cast loses integer precision}}
  return (int) ll;
  return (short) ll;
  return (char) ll;
  return (long long) BIG;  // expected-warning {{implicit cast loses integer precision}}
  return (long) BIG; // expected-warning {{implicit cast loses integer precision}}
  return (int) BIG;
  return (short) BIG;
  return (char) BIG;
}

long test4(long long ll) {
  return (long long) ll;
  return (long) ll;
  return (int) ll;
  return (short) ll;
  return (char) ll;
  return (long long) BIG;
  return (long) BIG;
  return (int) BIG;
  return (short) BIG;
  return (char) BIG;
}

long long test5(long long ll) {
  return (long long) ll;
  return (long) ll;
  return (int) ll;
  return (short) ll;
  return (char) ll;
  return (long long) BIG;
  return (long) BIG;
  return (int) BIG;
  return (short) BIG;
  return (char) BIG;
}

void takes_char(char);
void takes_short(short);
void takes_int(int);
void takes_long(long);
void takes_longlong(long long);
void takes_float(float);
void takes_double(double);
void takes_longdouble(long double);

void test6(char v) {
  takes_char(v);
  takes_short(v);
  takes_int(v);
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test7(short v) {
  takes_char(v); // expected-warning {{implicit cast loses integer precision}}
  takes_short(v);
  takes_int(v);
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test8(int v) {
  takes_char(v); // expected-warning {{implicit cast loses integer precision}}
  takes_short(v); // expected-warning {{implicit cast loses integer precision}}
  takes_int(v);
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test9(long v) {
  takes_char(v); // expected-warning {{implicit cast loses integer precision}}
  takes_short(v); // expected-warning {{implicit cast loses integer precision}}
  takes_int(v); // expected-warning {{implicit cast loses integer precision}}
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test10(long long v) {
  takes_char(v); // expected-warning {{implicit cast loses integer precision}}
  takes_short(v); // expected-warning {{implicit cast loses integer precision}}
  takes_int(v); // expected-warning {{implicit cast loses integer precision}}
  takes_long(v);
  takes_longlong(v);
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test11(float v) {
  takes_char(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_short(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_int(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_long(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_longlong(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_float(v);
  takes_double(v);
  takes_longdouble(v);
}

void test12(double v) {
  takes_char(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_short(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_int(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_long(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_longlong(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_float(v); // expected-warning {{implicit cast loses floating-point precision}}
  takes_double(v);
  takes_longdouble(v);
}

void test13(long double v) {
  takes_char(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_short(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_int(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_long(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_longlong(v); // expected-warning {{implicit cast turns floating-point number into integer}}
  takes_float(v); // expected-warning {{implicit cast loses floating-point precision}}
  takes_double(v); // expected-warning {{implicit cast loses floating-point precision}}
  takes_longdouble(v);
}

void test14(long l) {
  // Fine because of the boolean whitelist.
  char c;
  c = (l == 4);
  c = ((l <= 4) && (l >= 0));
  c = ((l <= 4) && (l >= 0)) || (l > 20);
}

void test15(char c) {
  c = c + 1 + c * 2;
  c = (short) c + 1 + c * 2; // expected-warning {{implicit cast loses integer precision}}
}

// PR 5422
extern void *test16_external;
void test16(void) {
  int a = (unsigned long) &test16_external; // expected-warning {{implicit cast loses integer precision}}
}
