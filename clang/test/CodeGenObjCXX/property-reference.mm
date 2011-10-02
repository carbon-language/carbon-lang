// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fobjc-fragile-abi -emit-llvm -o - | FileCheck %s
// rdar://9208606

struct MyStruct {
  int x;
  int y;
  int z;
};

@interface MyClass {
  MyStruct _foo;
}

@property (assign, readwrite) const MyStruct& foo;

- (const MyStruct&) foo;
- (void) setFoo:(const MyStruct&)inFoo;
@end

void test0() {
  MyClass* myClass;
  MyStruct myStruct;

  myClass.foo = myStruct;

  const MyStruct& currentMyStruct = myClass.foo;   
}

// CHECK: [[C:%.*]] = call %struct.MyStruct* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK:   store %struct.MyStruct* [[C]], %struct.MyStruct** [[D:%.*]]

namespace test1 {
  struct A { A(); A(const A&); A&operator=(const A&); ~A(); };
}
@interface Test1 {
  test1::A ivar;
}
@property const test1::A &prop1;
@end
@implementation Test1
@synthesize prop1 = ivar;
@end
// CHECK:    define internal [[A:%.*]]* @"\01-[Test1 prop1]"(
// CHECK:      [[SELF:%.*]] = alloca [[TEST1:%.*]]*, align 8
// CHECK:      [[T0:%.*]] = load [[TEST1]]** [[SELF]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST1]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i8* [[T1]], i64 0
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT: ret [[A]]* [[T3]]

// CHECK:    define internal void @"\01-[Test1 setProp1:]"(
// CHECK:      call [[A]]* @_ZN5test11AaSERKS0_(
// CHECK-NEXT: ret void

