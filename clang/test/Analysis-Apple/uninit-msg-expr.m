// RUN: clang -checker-simple -verify %s

#include <Foundation/NSString.h>
#include <Foundation/NSArray.h>

unsigned f1() {
  NSString *aString;
  return [aString length]; // expected-warning {{Receiver in message expression is an uninitialized value}}
}

unsigned f2() {
  NSString *aString = nil;
  return [aString length]; // no-warning
}

void f3() {
  NSMutableArray *aArray = [NSArray array];
  NSString *aString;
  [aArray addObject:aString]; // expected-warning {{Pass-by-value argument in message expression is undefined.}}
}
