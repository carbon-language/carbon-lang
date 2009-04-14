// RUN: clang-cc -fsyntax-only -verify %s

@class A, B, C;

void f() {
  goto L; // expected-error{{illegal jump}}
  goto L2; // expected-error{{illegal jump}}
  goto L3; // expected-error{{illegal jump}}
  @try {
L: ;
  } @catch (A *x) {
L2: ;
  } @catch (B *x) {
  } @catch (C *c) {
  } @finally {
L3: ;
  }
}
