// RUN: %clang_cc1 -emit-llvm -o - %s

struct CGRect {
  char* origin;
  unsigned size;
};
typedef struct CGRect CGRect;

extern "C" bool CGRectIsEmpty(CGRect);

@interface Foo {
  CGRect out;
}
@property CGRect bounds;
- (CGRect) out;
@end


@implementation Foo

- (void)bar {
    CGRect dataRect;
    CGRect virtualBounds;

  dataRect = CGRectIsEmpty(virtualBounds) ? self.bounds : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? [self bounds] : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? virtualBounds : self.bounds;

  dataRect = CGRectIsEmpty(virtualBounds) ? self.out : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? [self out] : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? virtualBounds : self.out;
}

@dynamic bounds;
- (CGRect) out { return out; }
@end
