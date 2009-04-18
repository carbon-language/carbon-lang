// RUN: clang-cc -fsyntax-only -verify %s

int test1(int x) {
  goto L;    // expected-error{{illegal goto into protected scope}}
  int a[x];  // expected-note {{jump bypasses initialization of variable length array}}
  int b[x];  // expected-note {{jump bypasses initialization of variable length array}}
  L:
  return sizeof a;
}

int test2(int x) {
  goto L;            // expected-error{{illegal goto into protected scope}}
  typedef int a[x];  // expected-note {{jump bypasses initialization of VLA typedef}}
  L:
  return sizeof(a);
}

void test3clean(int*);

int test3() {
  goto L;            // expected-error{{illegal goto into protected scope}}
int a __attribute((cleanup(test3clean))); // expected-note {{jump bypasses initialization of declaration with __attribute__((cleanup))}}
L:
  return a;
}

int test4(int x) {
  goto L;       // expected-error{{illegal goto into protected scope}}
int a[x];       // expected-note {{jump bypasses initialization of variable length array}}
  test4(x);
L:
  return sizeof a;
}

int test5(int x) {
  int a[x];
  test5(x);
  goto L;  // Ok.
L:
  goto L;  // Ok.
  return sizeof a;
}

int test6() { 
  // just plain invalid.
  goto x;  // expected-error {{use of undeclared label 'x'}}
}

void test7(int x) {
foo:
  switch (x) {      // expected-error {{illegal switch into protected scope}}
  case 1: ;
    int a[x];       // expected-note {{jump bypasses initialization of variable length array}}
  case 2:
    a[1] = 2;
    break;
  }
}


// FIXME: Switch cases etc.
