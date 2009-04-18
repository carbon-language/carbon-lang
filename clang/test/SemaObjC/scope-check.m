// RUN: clang-cc -fsyntax-only -verify %s

@class A, B, C;

void test1() {
  goto L; // expected-error{{illegal goto into protected scope}}
  goto L2; // expected-error{{illegal goto into protected scope}}
  goto L3; // expected-error{{illegal goto into protected scope}}
  @try {   // expected-note {{jump bypasses initialization of @try block}}
L: ;
  } @catch (A *x) { // expected-note {{jump bypasses initialization of @catch block}}
L2: ;
  } @catch (B *x) {
  } @catch (C *c) {
  } @finally {// expected-note {{jump bypasses initialization of @finally block}}
L3: ;
  }
  
  @try {
    goto L4; // expected-error{{illegal goto into protected scope}}
    goto L5; // expected-error{{illegal goto into protected scope}}
  } @catch (C *c) { // expected-note {{jump bypasses initialization of @catch block}}
  L5: ;
    goto L6; // expected-error{{illegal goto into protected scope}}
  } @catch (B *c) { // expected-note {{jump bypasses initialization of @catch block}}
  L6: ;
  } @finally { // expected-note {{jump bypasses initialization of @finally block}}
  L4: ;
  }
 
  
  @try { // expected-note 2 {{jump bypasses initialization of @try block}}
  L7: ;
  } @catch (C *c) {
    goto L7; // expected-error{{illegal goto into protected scope}}
  } @finally {
    goto L7; // expected-error{{illegal goto into protected scope}}
  }
  
  goto L8;  // expected-error{{illegal goto into protected scope}}
  @try { 
  } @catch (A *c) {
  } @catch (B *c) {
  } @catch (C *c) { // expected-note {{jump bypasses initialization of @catch block}}
  L8: ;
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

@interface Greeter
+ (void) hello;
@end

@implementation Greeter
+ (void) hello {

  @try {
    goto blargh;     // expected-error {{illegal goto into protected scope}}
  } @catch (...) {   // expected-note {{jump bypasses initialization of @catch block}}
  blargh: ;
  }
}
@end
