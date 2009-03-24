// RUN: clang-cc -verify %s

@protocol P;

void f() {
  @try {
  } @catch (void a) { // expected-error{{@catch parameter is not an Objective-C class type}}
  } @catch (int) { // expected-error{{@catch parameter is not an Objective-C class type}}
  } @catch (int *b) { // expected-error{{@catch parameter is not an Objective-C class type}}
  } @catch (id <P> c) { // expected-error{{illegal qualifiers on @catch parameter}}
  }
}

