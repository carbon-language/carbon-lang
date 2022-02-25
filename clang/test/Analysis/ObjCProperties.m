// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -analyzer-store=region -Wno-objc-root-class %s -verify
// expected-no-diagnostics

// The point of this test cases is to exercise properties in the static
// analyzer

@interface MyClass {
@private
    id _X;
}
- (id)initWithY:(id)Y;
@property(copy, readwrite) id X;
@end

@implementation MyClass
@synthesize X = _X;
- (id)initWithY:(id)Y {
  self.X = Y;
  return self;
}
@end
