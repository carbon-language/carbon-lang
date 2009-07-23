// RUN: clang-cc -analyze -warn-security-syntactic %s -verify

// <rdar://problem/6336718> rule request: floating point used as loop 
//  condition (FLP30-C, FLP-30-CPP)
//
// For reference: https://www.securecoding.cert.org/confluence/display/seccode/FLP30-C.+Do+not+use+floating+point+variables+as+loop+counters
//
void test_float_condition() {
  for (float x = 0.1f; x <= 1.0f; x += 0.1f) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (float x = 100000001.0f; x <= 100000010.0f; x += 1.0f) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (float x = 100000001.0f; x <= 100000010.0f; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (double x = 100000001.0; x <= 100000010.0; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  for (double x = 100000001.0; ((x)) <= 100000010.0; ((x))++ ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  for (double x = 100000001.0; 100000010.0 >= x; x = x + 1.0 ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  int i = 0;
  for (double x = 100000001.0; ((x)) <= 100000010.0; ((x))++, ++i ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  typedef float FooType;
  for (FooType x = 100000001.0f; x <= 100000010.0f; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'FooType'}}
}

