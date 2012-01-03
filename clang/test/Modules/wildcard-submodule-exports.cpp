// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fmodules -I %S/Inputs/wildcard-submodule-exports %s -verify

__import_module__ C.One;

void test_C_One() {
  int *A1_ptr = A1;
  long *C1_ptr = C1;
  (void)B1; // expected-error{{use of undeclared identifier 'B1'}}
}

__import_module__ C.Two;

void test_C_Two() {
  unsigned int *A2_ptr = A2;
  unsigned short *B2_ptr = B2;
  unsigned long *C2_ptr = C2;
}

__import_module__ B.One;

void test_B_One() {
  short *B1_ptr = B1;
}

