// RUN: clang-cc -fsyntax-only -verify %s

@class A, B, C;

void test1() {
  goto L; // expected-error{{illegal goto into protected scope}}
  goto L2; // expected-error{{illegal goto into protected scope}}
  goto L3; // expected-error{{illegal goto into protected scope}}
  @try {   // expected-note 3 {{jump bypasses initialization of @try block}}
L: ;
  } @catch (A *x) {
L2: ;
  } @catch (B *x) {
  } @catch (C *c) {
  } @finally {
L3: ;
  }
}

void test2(int a) {
  if (a) goto L0;
  @try {} @finally {}
 L0:
  return;
}

// rdar://6803963
void test3() {
  @try {
    goto blargh;
  blargh: ;
  } @catch (...) {}
}
