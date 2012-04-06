// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core -analyzer-store=region -analyzer-constraints=basic -Wno-objc-root-class %s -verify
// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core -analyzer-store=region -analyzer-constraints=range -Wno-objc-root-class %s -verify

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
