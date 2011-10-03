// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fobjc-fragile-abi -emit-llvm -o - | FileCheck %s
// rdar://10188258

struct Foo {int i;};

@interface ObjCTest  { }
@property (nonatomic, readonly) Foo& FooRefProperty;
@end


@implementation ObjCTest
@dynamic FooRefProperty;

-(void) test {
    Foo& f = self.FooRefProperty;
}
@end

// CHECK: [[T0:%.*]] = load {{%.*}} [[S0:%.*]]
// CHECK: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
// CHECK:  [[T2:%.*]]  = bitcast {{%.*}} [[T0]] to i8*
// CHECK:  @objc_msgSend

