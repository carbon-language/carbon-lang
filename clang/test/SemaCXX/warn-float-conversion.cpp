// RUN: %clang_cc1 -verify -fsyntax-only %s -Wfloat-conversion

bool ReturnBool(float f) {
  return f;  //expected-warning{{conversion}}
}

char ReturnChar(float f) {
  return f;  //expected-warning{{conversion}}
}

int ReturnInt(float f) {
  return f;  //expected-warning{{conversion}}
}

long ReturnLong(float f) {
  return f;  //expected-warning{{conversion}}
}

void Convert(float f, double d, long double ld) {
  bool b;
  char c;
  int i;
  long l;

  b = f;  //expected-warning{{conversion}}
  b = d;  //expected-warning{{conversion}}
  b = ld;  //expected-warning{{conversion}}
  c = f;  //expected-warning{{conversion}}
  c = d;  //expected-warning{{conversion}}
  c = ld;  //expected-warning{{conversion}}
  i = f;  //expected-warning{{conversion}}
  i = d;  //expected-warning{{conversion}}
  i = ld;  //expected-warning{{conversion}}
  l = f;  //expected-warning{{conversion}}
  l = d;  //expected-warning{{conversion}}
  l = ld;  //expected-warning{{conversion}}
}

