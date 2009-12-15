// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

struct test1_D {
  double d;
} d1;

void test1() {
  throw d1;
}

// CHECK:     define void @_Z5test1v() nounwind {
// CHECK-NEXT:entry:
// CHECK-NEXT:  %exception.ptr = alloca i8*
// CHECK-NEXT:  %exception = call i8* @__cxa_allocate_exception(i64 8)
// CHECK-NEXT:  store i8* %exception, i8** %exception.ptr
// CHECK-NEXT:  %0 = bitcast i8* %exception to %struct.test1_D*
// CHECK-NEXT:  %tmp = bitcast %struct.test1_D* %0 to i8*
// CHECK-NEXT:  call void @llvm.memcpy.i64(i8* %tmp, i8* bitcast (%struct.test1_D* @d1 to i8*), i64 8, i32 8)
// CHECK-NEXT:  call void @__cxa_throw(i8* %exception, i8* bitcast (%0* @_ZTI7test1_D to i8*), i8* null) noreturn
// CHECK-NEXT:  unreachable


struct test2_D {
  test2_D(const test2_D&o);
  test2_D();
  virtual void bar() { }
  int i; int j;
} d2;

void test2() {
  throw d2;
}

// CHECK:     define void @_Z5test2v() nounwind {
// CHECK-NEXT:entry:
// CHECK-NEXT:  %exception.ptr = alloca i8*
// CHECK-NEXT:  %exception = call i8* @__cxa_allocate_exception(i64 16)
// CHECK-NEXT:  store i8* %exception, i8** %exception.ptr
// CHECK-NEXT:  %0 = bitcast i8* %exception to %struct.test2_D*
// CHECK:       invoke void @_ZN7test2_DC1ERKS_(%struct.test2_D* %0, %struct.test2_D* @d2)
// CHECK-NEXT:     to label %invoke.cont unwind label %terminate.handler
// CHECK:  call void @__cxa_throw(i8* %exception, i8* bitcast (%0* @_ZTI7test2_D to i8*), i8* null) noreturn
// CHECK-NEXT:  unreachable


struct test3_D {
  test3_D() { }
  test3_D(volatile test3_D&o);
  virtual void bar();
};

void test3() {
  throw (volatile test3_D *)0;
}

// CHECK:     define void @_Z5test3v() nounwind {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %exception.ptr = alloca i8*
// CHECK-NEXT:   %exception = call i8* @__cxa_allocate_exception(i64 8)
// CHECK-NEXT:   store i8* %exception, i8** %exception.ptr
// CHECK-NEXT:   %0 = bitcast i8* %exception to %struct.test3_D**
// CHECK-NEXT:   store %struct.test3_D* null, %struct.test3_D** %0
// CHECK-NEXT:   call void @__cxa_throw(i8* %exception, i8* bitcast (%1* @_ZTIPV7test3_D to i8*), i8* null) noreturn
// CHECK-NEXT:   unreachable


void test4() {
  throw;
}

// CHECK:     define void @_Z5test4v() nounwind {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__cxa_rethrow() noreturn
// CHECK-NEXT:   unreachable
