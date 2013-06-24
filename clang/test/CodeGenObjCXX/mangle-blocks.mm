// RUN: %clang_cc1 -emit-llvm -fblocks -o - -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck %s

// CHECK: @_ZGVN23__block_prefix_internal5valueE = internal global i64 0
// CHECK: @_ZN24__block_prefix_internal35namebE = internal global i8*

int f();

void foo() {
  // CHECK: define internal i32 @___Z3foov_block_invoke
  // CHECK: call i32 @__cxa_guard_acquire(i64* @_ZGVN23__block_prefix_internal5valueE
  (void)^(int x) { 
    static int value = f();
    return x + value;
  };
}

// CHECK: define internal i32 @i_block_invoke
int i = ^(int x) { return x;}(i);

@interface A
- (void)method;
@end

@implementation A
- (void)method { 
  // CHECK: define internal signext i8 @"__11-[A method]_block_invoke"
  (void)^(int x) {
    // CHECK: @_ZN24__block_prefix_internal04nameE
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
  // CHECK: define internal signext i8 @___Z3fooi_block_invoke
  void bar() {
    (void)^(int x) { 
      // CHECK: @_ZN24__block_prefix_internal14nameE
      static const char *name = "hello";
      return name[x];
    };
  }
}

class C {
  C();
};
C::C() {
  (void)^(int x) { 
    // CHECK: @_ZN24__block_prefix_internal35namebE
    static const char *nameb = "hello";
    return nameb[x];
  };
}
