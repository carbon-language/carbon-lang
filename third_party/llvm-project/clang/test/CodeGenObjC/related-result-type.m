// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

@interface NSObject
+ (id)alloc;
- (id)init;
- (id)retain;
@end

@interface NSString : NSObject
@end

// CHECK-LABEL: define {{.*}}void @test1()
void test1() {
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: bitcast i8*
  NSString *str1 = [[[NSString alloc] init] retain];
}

// CHECK-LABEL: define {{.*}}void @test2()
void test2() {
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: {{call.*@objc_msgSend}}
  // CHECK: bitcast i8*
  NSString *str1 = NSString.alloc.init.retain;
}

@interface Test2 : NSString
- (id)init;
@end

@implementation Test2
// CHECK: define internal {{.*}}i8* @"\01-[Test2 init]"
- (id)init {
  // CHECK: {{call.*@objc_msgSendSuper}}
  // CHECK-NEXT: bitcast i8*
  return [super init];
}
@end

@interface Test3 : NSString
- (id)init;
@end

@implementation Test3
// CHECK: define internal {{.*}}i8* @"\01-[Test3 init]"
- (id)init {
  // CHECK: {{call.*@objc_msgSendSuper}}
  // CHECK-NEXT: bitcast i8*
  return [super init];
}
@end
