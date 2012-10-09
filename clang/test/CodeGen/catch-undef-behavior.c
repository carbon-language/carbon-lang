// RUN: %clang_cc1 -fcatch-undefined-behavior -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

// CHECK: @[[INT_STR:.*]] = private unnamed_addr constant [6 x i8] c"'int'\00"
// CHECK: @[[INT:.*]] = private unnamed_addr constant { i8*, i16, i16 } { i8* getelementptr inbounds ([6 x i8]* @[[INT_STR]], i32 0, i32 0), i16 0, i16 11 }

// FIXME: When we only emit each type once, use [[INT]] more below.
// CHECK: @[[LINE_100:.*]] = private unnamed_addr constant {{.*}}, i32 100, i32 5 {{.*}} @[[INT]], i64 4, i8 1
// CHECK: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 10 {{.*}}, i64 4, i8 0
// CHECK: @[[LINE_300_A:.*]] = {{.*}}, i32 300, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_300_B:.*]] = {{.*}}, i32 300, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 10 {{.*}} @{{.*}}, i64 4, i8 0 }
// CHECK: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 3 {{.*}} @{{.*}}, i64 4, i8 1 }

// CHECK: @[[STRUCT_S_STR:.*]] = private unnamed_addr constant [11 x i8] c"'struct S'\00"
// CHECK: @[[STRUCT_S:.*]] = private unnamed_addr constant {{.*}}@[[STRUCT_S_STR]], i32 0, i32 0), i16 -1, i16 0 }

// CHECK: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 14 {{.*}} @[[STRUCT_S]], i64 4, i8 3 }
// CHECK: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 12 {{.*}} @{{.*}} }

// PR6805
// CHECK: @foo
void foo() {
  union { int i; } u;
  // CHECK:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64({{.*}} %[[PTR:.*]], i1 false)
  // CHECK-NEXT: %[[CHECK1:.*]] = icmp uge i64 %[[SIZE]], 4

  // CHECK:      %[[PTRTOINT:.*]] = ptrtoint {{.*}} %[[PTR]] to i64
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRTOINT]], 3
  // CHECK-NEXT: %[[CHECK2:.*]] = icmp eq i64 %[[MISALIGN]], 0

  // CHECK:      %[[OK:.*]] = and i1 %[[CHECK1]], %[[CHECK2]]
  // CHECK-NEXT: br i1 %[[OK]]

  // CHECK:      %[[ARG:.*]] = ptrtoint {{.*}} %[[PTR]] to i64
  // CHECK-NEXT: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_100]] to i8*), i64 %[[ARG]]) noreturn nounwind
#line 100
  u.i=1;
}

// CHECK: @bar
int bar(int *a) {
  // CHECK:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK:      %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // CHECK:      %[[ARG:.*]] = ptrtoint
  // CHECK-NEXT: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_200]] to i8*), i64 %[[ARG]]) noreturn nounwind
#line 200
  return *a;
}

// CHECK: @lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK:      %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[INBOUNDS]]

  // FIXME: Only emit one trap block here.
  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_300_A]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind

  // CHECK:      %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]
  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT]], 0
  // CHECK-NEXT: br i1 %[[NO_OVERFLOW]]

  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_300_B]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind

  // CHECK:      %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
#line 300
  return a << b;
}

// CHECK: @rsh_inbounds
int rsh_inbounds(int a, int b) {
  // CHECK:      %[[INBOUNDS:.*]] = icmp ult i32 %[[RHS:.*]], 32
  // CHECK:      br i1 %[[INBOUNDS]]

  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_400]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind

  // CHECK:      %[[RET:.*]] = ashr i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
#line 400
  return a >> b;
}

// CHECK: @load
int load(int *p) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_500]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 500
  return *p;
}

// CHECK: @store
void store(int *p, int q) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_600]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 600
  *p = q;
}

struct S { int k; };

// CHECK: @member_access
int *member_access(struct S *p) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_700]] to i8*), i64 %{{.*}}) noreturn nounwind
#line 700
  return &p->k;
}

// CHECK: @signed_overflow
int signed_overflow(int a, int b) {
  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_add_overflow(i8* bitcast ({{.*}} @[[LINE_800]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]]) noreturn nounwind
#line 800
  return a + b;
}

// CHECK: @no_return
int no_return() {
  // Reaching the end of a noreturn function is fine in C.
  // CHECK-NOT: call
  // CHECK-NOT: unreachable
  // CHECK: ret i32
}
