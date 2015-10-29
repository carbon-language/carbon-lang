// RUN: %clang_cc1 -verify -fsyntax-only %s -Wdouble-promotion

float ReturnFloatFromDouble(double d) {
  return d;
}

float ReturnFloatFromLongDouble(long double ld) {
  return ld;
}

double ReturnDoubleFromLongDouble(long double ld) {
  return ld;
}

double ReturnDoubleFromFloat(float f) {
  return f;  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
}

long double ReturnLongDoubleFromFloat(float f) {
  return f;  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
}

long double ReturnLongDoubleFromDouble(double d) {
  return d;  //expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
}

void Convert(float f, double d, long double ld) {
  d = f;  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  ld = f; //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  ld = d; //expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  f = d;
  f = ld;
  d = ld;
}
