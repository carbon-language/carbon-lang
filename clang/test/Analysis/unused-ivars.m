// RUN: clang-cc -analyze -warn-objc-unused-ivars %s -verify

@interface A
{
  @private int x; // expected-warning {{Instance variable 'x' in class 'A' is never used}}
}
@end

@implementation A @end

