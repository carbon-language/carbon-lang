// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=inlining -analyzer-output=text -verify %s

void zero(int **p) {
  *p = 0;
  // expected-note@-1 {{Null pointer value stored to 'a'}}
}

void testZero(int *a) {
  zero(&a);
  // expected-note@-1 {{Calling 'zero'}}
  // expected-note@-2 {{Returning from 'zero'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}


void check(int *p) {
  if (p) {
    // expected-note@-1 + {{Assuming 'p' is null}}
    // expected-note@-2 + {{Assuming pointer value is null}}
    // expected-note@-3 + {{Taking false branch}}
    return;
  }
  return;
}

void testCheck(int *a) {
  check(a);
  // expected-note@-1 {{Calling 'check'}}
  // expected-note@-2 {{Returning from 'check'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}


int *getPointer();

void testInitCheck() {
  int *a = getPointer();
  // expected-note@-1 {{Variable 'a' initialized here}}
  check(a);
  // expected-note@-1 {{Calling 'check'}}
  // expected-note@-2 {{Returning from 'check'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}

void testStoreCheck(int *a) {
  a = getPointer();
  // expected-note@-1 {{Value assigned to 'a'}}
  check(a);
  // expected-note@-1 {{Calling 'check'}}
  // expected-note@-2 {{Returning from 'check'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}
