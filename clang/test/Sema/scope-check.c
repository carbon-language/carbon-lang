// RUN: clang -fsyntax-only -verify %s

int test1(int x) {
  goto L; // expected-error{{illegal jump}}
  int a[x];
  L:
  return sizeof a;
}
