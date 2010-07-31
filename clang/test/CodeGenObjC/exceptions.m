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

// Test that modifications to local variables are respected under
// optimization.  rdar://problem/8160285

// CHECK: define i32 @f2()
int f2() {
  extern void foo(void);

  // CHECK:        [[X:%.*]] = alloca i32
  // CHECK:        store i32 0, i32* [[X]]
  int x = 0;

  // CHECK:        [[SETJMP:%.*]] = call i32 @_setjmp
  // CHECK-NEXT:   [[CAUGHT:%.*]] = icmp eq i32 [[SETJMP]], 0
  // CHECK-NEXT:   br i1 [[CAUGHT]]
  @try {
    // This load should be coalescable with the store of 0, so if the
    // optimizers ever figure out that out, that's okay.
    // CHECK:      [[T1:%.*]] = load i32* [[X]]
    // CHECK-NEXT: [[T2:%.*]] = add nsw i32 [[T1]], 1
    // CHECK-NEXT: store i32 [[T2]], i32* [[X]]
    x++;
    // CHECK-NEXT: call void asm sideeffect "", "*m"(i32* [[X]]) nounwind
    // CHECK-NEXT: call void @foo()
    foo();
  } @catch (id) {
    // Landing pad.  It turns out that the re-enter is unnecessary here.
    // CHECK:      call void asm sideeffect "", "=*m"(i32* [[X]]) nounwind
    // CHECK-NEXT: call i8* @objc_exception_extract
    // CHECK-NEXT: call void @objc_exception_try_enter
    // CHECK-NEXT: call i32 @_setjmp
    // CHECK-NEXT: icmp eq i32
    // CHECK-NEXT: br i1

    // Catch handler.
    // CHECK:      [[T1:%.*]] = load i32* [[X]]
    // CHECK-NEXT: [[T2:%.*]] = add nsw i32 [[T1]], -1
    // CHECK-NEXT: store i32 [[T2]], i32* [[X]]
    // CHECK-NEXT: br label
    x--;
  }
  // CHECK:        call void @objc_exception_try_exit
  // CHECK-NEXT:   [[T:%.*]] = load i32* [[X]]
  // CHECK:        ret i32 [[T]]
  return x;
}
