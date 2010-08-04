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
    // CHECK:      call void asm sideeffect "", "*m"
    // CHECK-NEXT: call void @foo()
      foo();
    // CHECK-NEXT: call void @objc_exception_try_exit
    // CHECK-NEXT: ret void

    // CHECK:      call void asm sideeffect "", "=*m"
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
  // CHECK:        store i32 5, i32* [[X]]
  int x = 0;
  x += 5;

  // CHECK:        [[SETJMP:%.*]] = call i32 @_setjmp
  // CHECK-NEXT:   [[CAUGHT:%.*]] = icmp eq i32 [[SETJMP]], 0
  // CHECK-NEXT:   br i1 [[CAUGHT]]
  @try {
    // If the optimizers ever figure out how to make this store 6,
    // that's okay.
    // CHECK:      [[T1:%.*]] = load i32* [[X]]
    // CHECK-NEXT: [[T2:%.*]] = add nsw i32 [[T1]], 1
    // CHECK-NEXT: store i32 [[T2]], i32* [[X]]
    x++;
    // CHECK-NEXT: call void asm sideeffect "", "*m,*m"(i32* [[X]]
    // CHECK-NEXT: call void @foo()
    // CHECK-NEXT: call void @objc_exception_try_exit
    // CHECK-NEXT: [[T:%.*]] = load i32* [[X]]
    // CHECK-NEXT: ret i32 [[T]]
    foo();
  } @catch (id) {
    // Landing pad.  Note that we elide the re-enter.
    // CHECK:      call void asm sideeffect "", "=*m,=*m"(i32* [[X]]
    // CHECK-NEXT: call i8* @objc_exception_extract
    // CHECK-NEXT: [[T1:%.*]] = load i32* [[X]]
    // CHECK-NEXT: [[T2:%.*]] = add nsw i32 [[T1]], -1

    // This store is dead.
    // CHECK-NEXT: store i32 [[T2]], i32* [[X]]

    // CHECK-NEXT: ret i32 [[T2]]
    x--;
  }
  return x;
}
