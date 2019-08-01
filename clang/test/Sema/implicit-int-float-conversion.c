// RUN: %clang_cc1 %s -verify -Wno-conversion -Wimplicit-int-float-conversion

long testReturn(long a, float b) {
  return a + b; // expected-warning {{implicit conversion from 'long' to 'float' may lose precision}}
}

void testAssignment() {
  float f = 222222;
  double b = 222222222222L;

#ifndef __ILP32__
  float ff = 222222222222L;    // expected-warning {{implicit conversion from 'long' to 'float' changes value from 222222222222 to 222222221312}}
  float ffff = 222222222222UL; // expected-warning {{implicit conversion from 'unsigned long' to 'float' changes value from 222222222222 to 222222221312}}
#else
  float ff = 222222222222L;    // expected-warning {{implicit conversion from 'long long' to 'float' changes value from 222222222222 to 222222221312}}
  float ffff = 222222222222UL; // expected-warning {{implicit conversion from 'unsigned long long' to 'float' changes value from 222222222222 to 222222221312}}
#endif

  long l = 222222222222L;
  float fff = l; // expected-warning {{implicit conversion from 'long' to 'float' may lose precision}}
}

void testExpression() {
  float a = 0.0f;

#ifndef __ILP32__
  float b = 222222222222L + a; // expected-warning {{implicit conversion from 'long' to 'float' changes value from 222222222222 to 222222221312}}
#else
  float b = 222222222222L + a; // expected-warning {{implicit conversion from 'long long' to 'float' changes value from 222222222222 to 222222221312}}
#endif

  float g = 22222222 + 22222222;
  float c = 22222222 + 22222223; // expected-warning {{implicit conversion from 'int' to 'float' changes value from 44444445 to 44444444}}

  int i = 0;
  float d = i + a; // expected-warning {{implicit conversion from 'int' to 'float' may lose precision}}

  double e = 0.0;
  double f = i + e;
}

void testCNarrowing() {
  // Since this is a C file. C++11 narrowing is not in effect.
  // In this case, we should issue warnings.
#ifndef __ILP32__
  float a = {222222222222L}; // expected-warning {{implicit conversion from 'long' to 'float' changes value from 222222222222 to 222222221312}}
#else
  float a = {222222222222L};   // expected-warning {{implicit conversion from 'long long' to 'float' changes value from 222222222222 to 222222221312}}
#endif

  long b = 222222222222L;
  float c = {b}; // expected-warning {{implicit conversion from 'long' to 'float' may lose precision}}
}
