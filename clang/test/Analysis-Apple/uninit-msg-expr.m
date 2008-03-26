// RUN: clang -grsimple -verify %s

#include <Foundation/NSString.h>

void f1() {
  NSString *aString;
  unsigned i = [aString length]; // expected-warning {{Receiver in message expression is an uninitialized value}}
}

void f2() {
  NSString *aString = nil;
  unsigned i = [aString length]; // no-warning
}
