// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

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

// CHECK: [[SRC:%.*]] = call %struct.CGRect bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK-NEXT:store %struct.CGRect [[SRC]], %struct.CGRect*
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
