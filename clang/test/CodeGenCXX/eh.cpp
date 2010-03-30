// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

struct test1_D {
  double d;
} d1;

void test1() {
  throw d1;
}

// CHECK:     define void @_Z5test1v() nounwind {
// CHECK:       %{{exception.ptr|1}} = alloca i8*
// CHECK-NEXT:  %{{exception|2}} = call i8* @__cxa_allocate_exception(i64 8)
// CHECK-NEXT:  store i8* %{{exception|2}}, i8** %{{exception.ptr|1}}
// CHECK-NEXT:  %{{0|3}} = bitcast i8* %{{exception|2}} to %struct.test1_D*
// CHECK-NEXT:  %{{tmp|4}} = bitcast %struct.test1_D* %{{0|3}} to i8*
// CHECK-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %{{tmp|4}}, i8* bitcast (%struct.test1_D* @d1 to i8*), i64 8, i32 8, i1 false)
// CHECK-NEXT:  call void @__cxa_throw(i8* %{{exception|2}}, i8* bitcast (%0* @_ZTI7test1_D to i8*), i8* null) noreturn
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
// CHECK:       %{{exception.ptr|1}} = alloca i8*
// CHECK-NEXT:  %{{exception|2}} = call i8* @__cxa_allocate_exception(i64 16)
// CHECK-NEXT:  store i8* %{{exception|2}}, i8** %{{\1}}
// CHECK-NEXT:  %{{0|3}} = bitcast i8* %{{exception|2}} to %struct.test2_D*
// CHECK:       invoke void @_ZN7test2_DC1ERKS_(%struct.test2_D* %{{0|3}}, %struct.test2_D* @d2)
// CHECK-NEXT:     to label %{{invoke.cont|8}} unwind label %{{terminate.handler|4}}
// CHECK:  call void @__cxa_throw(i8* %{{exception|2}}, i8* bitcast (%{{0|3}}* @_ZTI7test2_D to i8*), i8* null) noreturn
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
// CHECK:        %{{exception.ptr|1}} = alloca i8*
// CHECK-NEXT:   %{{exception|2}} = call i8* @__cxa_allocate_exception(i64 8)
// CHECK-NEXT:   store i8* %{{exception|2}}, i8** %{{exception.ptr|1}}
// CHECK-NEXT:   %{{0|3}} = bitcast i8* %{{exception|2}} to %struct.test3_D**
// CHECK-NEXT:   store %struct.test3_D* null, %struct.test3_D**
// CHECK-NEXT:   call void @__cxa_throw(i8* %{{exception|2}}, i8* bitcast (%1* @_ZTIPV7test3_D to i8*), i8* null) noreturn
// CHECK-NEXT:   unreachable


void test4() {
  throw;
}

// CHECK:     define void @_Z5test4v() nounwind {
// CHECK:        call void @__cxa_rethrow() noreturn
// CHECK-NEXT:   unreachable
