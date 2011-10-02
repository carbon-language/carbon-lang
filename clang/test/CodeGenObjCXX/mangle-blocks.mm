// RUN: %clang_cc1 -emit-llvm -fblocks -o - -triple x86_64-apple-darwin10 -fobjc-fragile-abi %s | FileCheck %s

// CHECK: @_ZGVN3foo20__foo_block_invoke_05valueE = internal global i64 0

int f();

void foo() {
  // CHECK: define internal i32 @__foo_block_invoke_0
  // CHECK: call i32 @__cxa_guard_acquire(i64* @_ZGVN3foo20__foo_block_invoke_05value
  (void)^(int x) { 
    static int value = f();
    return x + value;
  };
}

// CHECK: define internal i32 @__block_global_0
int i = ^(int x) { return x;}(i);

@interface A
- (void)method;
@end

@implementation A
- (void)method { 
  // CHECK: define internal signext i8 @"__11-[A method]_block_invoke_0"
  (void)^(int x) {
    // CHECK: @"_ZN11-[A method]30__11-[A method]_block_invoke_04nameE"
    static const char *name = "hello";
    return name[x];
  };
}
@end

void foo(int) {
  (void)^(int x) { 
    static const char *name = "hello";
    return name[x];
  };
}

namespace N {
  // CHECK: define internal signext i8 @__bar_block_invoke_0
  void bar() {
    (void)^(int x) { 
      // CHECK: @_ZN1N3bar20__bar_block_invoke_04nameE
      static const char *name = "hello";
      return name[x];
    };
  }
}
