// RUN: clang -grsimple -verify %s

#include <Foundation/NSString.h>
#include <Foundation/NSObjCRuntime.h>

NSComparisonResult f1(NSString* s) {
  NSString *aString = nil;
  return [s compare:aString]; // expected-warning {{Argument to NSString method 'compare:' cannot be nil.}}
}
