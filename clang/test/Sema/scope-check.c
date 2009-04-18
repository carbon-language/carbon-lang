// RUN: clang-cc -fsyntax-only -verify %s

/*
"/tmp/bug.c", line 2: error: transfer of control bypasses initialization of:
   variable length array "a" (declared at line 3)
   variable length array "b" (declared at line 3)
     goto L; 
     ^
"/tmp/bug.c", line 3: warning: variable "b" was declared but never referenced
     int a[x], b[x];
               ^
*/

int test1(int x) {
  goto L;    // expected-error{{illegal goto into protected scope}}
  int a[x];  // expected-note {{scope created by variable length array}}
  int b[x];  // expected-note {{scope created by variable length array}}
  L:
  return sizeof a;
}

int test2(int x) {
  goto L;            // expected-error{{illegal goto into protected scope}}
  typedef int a[x];  // expected-note {{scope created by VLA typedef}}
  L:
  return sizeof(a);
}

void test3clean(int*);

int test3() {
  goto L;            // expected-error{{illegal goto into protected scope}}
int a __attribute((cleanup(test3clean))); // expected-note {{scope created by declaration with __attribute__((cleanup))}}
L:
  return a;
}

int test4(int x) {
  goto L;       // expected-error{{illegal goto into protected scope}}
int a[x];       // expected-note {{scope created by variable length array}}
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


// FIXME: Switch cases etc.
