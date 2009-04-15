// RUN: clang-cc -fsyntax-only -verify %s

int test1(int x) {
  goto L; // expected-error{{illegal jump}}
  int a[x];
  L:
  return sizeof a;
}

int test2(int x) {
  goto L; // expected-error{{illegal jump}}
  typedef int a[x];
  L:
  return sizeof(a);
}

void test3clean(int*);

int test3() {
  goto L; // expected-error{{illegal jump}}
  int a __attribute((cleanup(test3clean)));
  L:
  return a;
}

int test4(int x) {
 goto L; // expected-error{{illegal jump}}
 int a[x];
 test4(x);
 L:
 return sizeof a;
}
