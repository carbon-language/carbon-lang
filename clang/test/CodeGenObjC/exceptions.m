// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -O2 -o - %s | FileCheck %s
//
// <rdar://problem/7471679> [irgen] [eh] Exception code built with clang (x86_64) crashes

// Just check that we don't emit any dead blocks.
@interface NSArray @end
void f0() {
  @try {
    @try {
      @throw @"a";
    } @catch(NSArray *e) {
    }
  } @catch (id e) {
  }
}

// CHECK: define void @f1()
void f1() {
  extern void foo(void);

  while (1) {
    // CHECK:      call void @objc_exception_try_enter
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call i32 @_setjmp(
    // CHECK-NEXT: icmp
    // CHECK-NEXT: br i1
    @try {
    // CHECK:      call void @foo()
      foo();
    // CHECK:      call void @objc_exception_try_exit
    // CHECK-NEXT: ret void

    // CHECK:      call i8* @objc_exception_extract
    // CHECK-NEXT: ret void
    } @finally {
      break;
    }
  }
}
