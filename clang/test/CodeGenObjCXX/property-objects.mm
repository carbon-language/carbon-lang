// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK: call void @_ZN1SC1ERKS_
// CHECK: call %class.S* @_ZN1SaSERKS_

class S {
public:
	S& operator = (const S&);
	S (const S&);
	S ();
};

@interface I {
  S position;
}
@property(assign, nonatomic) S position;
@end

@implementation I
@synthesize position;
@end
