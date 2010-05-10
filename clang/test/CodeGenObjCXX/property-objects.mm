// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK: call void @_ZN1SC1ERKS_
// CHECK: call %class.S* @_ZN1SaSERKS_
// CHECK: call %class.S* @_ZN6CGRectaSERKS_

class S {
public:
	S& operator = (const S&);
	S (const S&);
	S ();
};

struct CGRect {
	CGRect & operator = (const CGRect &);
};

@interface I {
  S position;
  CGRect bounds;
}
@property(assign, nonatomic) S position;
@property CGRect bounds;
- (void) initWithOwner;
@end

@implementation I
@synthesize position;
@synthesize bounds;
- (void)initWithOwner {
  CGRect labelLayerFrame = self.bounds;
  labelLayerFrame = self.bounds;
}
@end

int main() {
  I *i;
  S s1;
  i.position = s1;
  return 0;
}

