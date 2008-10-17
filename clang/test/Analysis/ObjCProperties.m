// RUN: clang -checker-simple %s -verify

// The point of this test cases is to exercise properties in the static
// analyzer

@interface MyClass {
@private
    id _X;
}
- (id)initWithY:(id)Y;
@property(copy, readonly) id X;
@end

@implementation MyClass
@synthesize X = _X;
- (id)initWithY:(id)Y {
  self.X = Y;
  return self;
}
@end
