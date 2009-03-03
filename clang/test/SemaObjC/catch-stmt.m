// RUN: clang -verify %s

void f() {
  @try {
  } @catch (void a) { // expected-error{{@catch parameter is not an Objective-C class type}}
  } @catch (int) { // expected-error{{@catch parameter is not an Objective-C class type}}
  } @catch (int *b) { // expected-error{{@catch parameter is not an Objective-C class type}}
  }
}

