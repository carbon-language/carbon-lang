// RUN: %clang_cc1 -fsanitize=alignment,null,object-size,shift,return,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fsanitize-undefined-trap-on-error -fsanitize=alignment,null,object-size,shift,return,signed-integer-overflow,vla-bound,float-cast-overflow,integer-divide-by-zero,bool,returns-nonnull-attribute,nonnull-attribute -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-TRAP
// RUN: %clang_cc1 -fsanitize=null -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-NULL
// RUN: %clang_cc1 -fsanitize=signed-integer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK-OVERFLOW

// CHECK: @[[INT:.*]] = private unnamed_addr constant { i16, i16, [6 x i8] } { i16 0, i16 11, [6 x i8] c"'int'\00" }

// FIXME: When we only emit each type once, use [[INT]] more below.
// CHECK: @[[LINE_100:.*]] = private unnamed_addr global {{.*}}, i32 100, i32 5 {{.*}} @[[INT]], i64 4, i8 1
// CHECK: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 10 {{.*}}, i64 4, i8 0
// CHECK: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 12 {{.*}} @{{.*}}, {{.*}} @{{.*}}
// CHECK: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 10 {{.*}} @{{.*}}, i64 4, i8 0 }
// CHECK: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 3 {{.*}} @{{.*}}, i64 4, i8 1 }

// CHECK: @[[STRUCT_S:.*]] = private unnamed_addr constant { i16, i16, [11 x i8] } { i16 -1, i16 0, [11 x i8] c"'struct S'\00" }

// CHECK: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 14 {{.*}} @[[STRUCT_S]], i64 4, i8 3 }
// CHECK: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 12 {{.*}} @{{.*}} }
// CHECK: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 11 {{.*}} @{{.*}} }

// CHECK-NULL: @[[LINE_100:.*]] = private unnamed_addr global {{.*}}, i32 100, i32 5 {{.*}}

// PR6805
// CHECK-LABEL: @foo
// CHECK-NULL-LABEL: @foo
// CHECK-TRAP-LABEL: @foo
void foo() {
  union { int i; } u;
  // CHECK:      %[[CHECK0:.*]] = icmp ne {{.*}}* %[[PTR:.*]], null
  // CHECK-TRAP: %[[CHECK0:.*]] = icmp ne {{.*}}* %[[PTR:.*]], null

  // CHECK:      %[[I8PTR:.*]] = bitcast i32* %[[PTR]] to i8*
  // CHECK-NEXT: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0i8(i8* %[[I8PTR]], i1 false)
  // CHECK-NEXT: %[[CHECK1:.*]] = icmp uge i64 %[[SIZE]], 4
  // CHECK-NEXT: %[[CHECK01:.*]] = and i1 %[[CHECK0]], %[[CHECK1]]

  // CHECK-TRAP:      %[[I8PTR:.*]] = bitcast i32* %[[PTR]] to i8*
  // CHECK-TRAP-NEXT: %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0i8(i8* %[[I8PTR]], i1 false)
  // CHECK-TRAP-NEXT: %[[CHECK1:.*]] = icmp uge i64 %[[SIZE]], 4
  // CHECK-TRAP-NEXT: %[[CHECK01:.*]] = and i1 %[[CHECK0]], %[[CHECK1]]

  // CHECK:      %[[PTRTOINT:.*]] = ptrtoint {{.*}}* %[[PTR]] to i64
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRTOINT]], 3
  // CHECK-NEXT: %[[CHECK2:.*]] = icmp eq i64 %[[MISALIGN]], 0

  // CHECK-TRAP:      %[[PTRTOINT:.*]] = ptrtoint {{.*}}* %[[PTR]] to i64
  // CHECK-TRAP-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRTOINT]], 3
  // CHECK-TRAP-NEXT: %[[CHECK2:.*]] = icmp eq i64 %[[MISALIGN]], 0

  // CHECK:      %[[OK:.*]] = and i1 %[[CHECK01]], %[[CHECK2]]
  // CHECK-NEXT: br i1 %[[OK]], {{.*}} !prof ![[WEIGHT_MD:.*]], !nosanitize

  // CHECK-TRAP:      %[[OK:.*]] = and i1 %[[CHECK01]], %[[CHECK2]]
  // CHECK-TRAP-NEXT: br i1 %[[OK]], {{.*}}

  // CHECK:      %[[ARG:.*]] = ptrtoint {{.*}} %[[PTR]] to i64
  // CHECK-NEXT: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_100]] to i8*), i64 %[[ARG]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW:#[0-9]+]]
  // CHECK-TRAP-NEXT: unreachable

  // With -fsanitize=null, only perform the null check.
  // CHECK-NULL: %[[NULL:.*]] = icmp ne {{.*}}, null
  // CHECK-NULL: br i1 %[[NULL]]
  // CHECK-NULL: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_100]] to i8*), i64 %{{.*}})
#line 100
  u.i=1;
}

// CHECK-LABEL: @bar
// CHECK-TRAP-LABEL: @bar
int bar(int *a) {
  // CHECK:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK-TRAP:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64
  // CHECK-TRAP-NEXT: icmp uge i64 %[[SIZE]], 4

  // CHECK:      %[[PTRINT:.*]] = ptrtoint
  // CHECK-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // CHECK-TRAP:      %[[PTRINT:.*]] = ptrtoint
  // CHECK-TRAP-NEXT: %[[MISALIGN:.*]] = and i64 %[[PTRINT]], 3
  // CHECK-TRAP-NEXT: icmp eq i64 %[[MISALIGN]], 0

  // CHECK:      %[[ARG:.*]] = ptrtoint
  // CHECK-NEXT: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_200]] to i8*), i64 %[[ARG]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable

#line 200
  return *a;
}

// CHECK-LABEL: @addr_space
int addr_space(int __attribute__((address_space(256))) *a) {
  // CHECK-NOT: __ubsan
  return *a;
}

// CHECK-LABEL: @lsh_overflow
// CHECK-TRAP-LABEL: @lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK:      %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[INBOUNDS]], label %[[CHECKBB:.*]], label %[[CONTBB:.*]]

  // CHECK-TRAP:      %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]], label %[[CHECKBB:.*]], label %[[CONTBB:.*]]

  // CHECK:      %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]
  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT]], 0
  // CHECK-NEXT: br label %[[CONTBB]]

  // CHECK-TRAP:      %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-TRAP-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]
  // CHECK-TRAP-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT]], 0
  // CHECK-TRAP-NEXT: br label %[[CONTBB]]

  // CHECK:      %[[VALID:.*]] = phi i1 [ %[[INBOUNDS]], {{.*}} ], [ %[[NO_OVERFLOW]], %[[CHECKBB]] ]
  // CHECK-NEXT: br i1 %[[VALID]], {{.*}} !prof ![[WEIGHT_MD]]

  // CHECK-TRAP:      %[[VALID:.*]] = phi i1 [ %[[INBOUNDS]], {{.*}} ], [ %[[NO_OVERFLOW]], %[[CHECKBB]] ]
  // CHECK-TRAP-NEXT: br i1 %[[VALID]]


  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_300]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]])
  // CHECK-NOT:  call void @__ubsan_handle_shift_out_of_bounds

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP:      unreachable
  // CHECK-TRAP-NOT:  call void @llvm.trap()

  // CHECK:      %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]

  // CHECK-TRAP:      %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-TRAP-NEXT: ret i32 %[[RET]]
#line 300
  return a << b;
}

// CHECK-LABEL: @rsh_inbounds
// CHECK-TRAP-LABEL: @rsh_inbounds
int rsh_inbounds(int a, int b) {
  // CHECK:      %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK:      br i1 %[[INBOUNDS]]

  // CHECK-TRAP: %[[INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-TRAP: br i1 %[[INBOUNDS]]

  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_shift_out_of_bounds(i8* bitcast ({{.*}} @[[LINE_400]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable

  // CHECK:      %[[RET:.*]] = ashr i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]

  // CHECK-TRAP:      %[[RET:.*]] = ashr i32 %[[LHS]], %[[RHS]]
  // CHECK-TRAP-NEXT: ret i32 %[[RET]]
#line 400
  return a >> b;
}

// CHECK-LABEL: @load
// CHECK-TRAP-LABEL: @load
int load(int *p) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_500]] to i8*), i64 %{{.*}})

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 500
  return *p;
}

// CHECK-LABEL: @store
// CHECK-TRAP-LABEL: @store
void store(int *p, int q) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_600]] to i8*), i64 %{{.*}})

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 600
  *p = q;
}

struct S { int k; };

// CHECK-LABEL: @member_access
// CHECK-TRAP-LABEL: @member_access
int *member_access(struct S *p) {
  // CHECK: call void @__ubsan_handle_type_mismatch(i8* bitcast ({{.*}} @[[LINE_700]] to i8*), i64 %{{.*}})

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 700
  return &p->k;
}

// CHECK-LABEL: @signed_overflow
// CHECK-TRAP-LABEL: @signed_overflow
int signed_overflow(int a, int b) {
  // CHECK:      %[[ARG1:.*]] = zext
  // CHECK-NEXT: %[[ARG2:.*]] = zext
  // CHECK-NEXT: call void @__ubsan_handle_add_overflow(i8* bitcast ({{.*}} @[[LINE_800]] to i8*), i64 %[[ARG1]], i64 %[[ARG2]])

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
#line 800
  return a + b;
}

// CHECK-LABEL: @no_return
// CHECK-TRAP-LABEL: @no_return
int no_return() {
  // Reaching the end of a noreturn function is fine in C.
  // FIXME: If the user explicitly requests -fsanitize=return, we should catch
  //        that here even though it's not undefined behavior.
  // CHECK-NOT: call
  // CHECK-NOT: unreachable
  // CHECK: ret i32

  // CHECK-TRAP-NOT: call
  // CHECK-TRAP-NOT: unreachable
  // CHECK-TRAP: ret i32
}

// CHECK-LABEL: @vla_bound
void vla_bound(int n) {
  // CHECK:      icmp sgt i32 %[[PARAM:.*]], 0
  //
  // CHECK:      %[[ARG:.*]] = zext i32 %[[PARAM]] to i64
  // CHECK-NEXT: call void @__ubsan_handle_vla_bound_not_positive(i8* bitcast ({{.*}} @[[LINE_900]] to i8*), i64 %[[ARG]])
#line 900
  int arr[n * 3];
}

// CHECK-LABEL: @int_float_no_overflow
float int_float_no_overflow(__int128 n) {
  // CHECK-NOT: call void @__ubsan_handle
  return n;
}

// CHECK-LABEL: @int_float_overflow
// CHECK-TRAP-LABEL: @int_float_overflow
float int_float_overflow(unsigned __int128 n) {
  // This is 2**104. FLT_MAX is 2**128 - 2**104.
  // CHECK: icmp ule i128 %{{.*}}, -20282409603651670423947251286016
  // CHECK: call void @__ubsan_handle_float_cast_overflow(

  // CHECK-TRAP: %[[INBOUNDS:.*]] = icmp ule i128 %{{.*}}, -20282409603651670423947251286016
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  return n;
}

// CHECK-LABEL: @int_fp16_overflow
// CHECK-TRAP-LABEL: @int_fp16_overflow
void int_fp16_overflow(int n, __fp16 *p) {
  // CHECK: %[[GE:.*]] = icmp sge i32 %{{.*}}, -65504
  // CHECK: %[[LE:.*]] = icmp sle i32 %{{.*}}, 65504
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(

  // CHECK-TRAP: %[[GE:.*]] = icmp sge i32 %{{.*}}, -65504
  // CHECK-TRAP: %[[LE:.*]] = icmp sle i32 %{{.*}}, 65504
  // CHECK-TRAP: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  *p = n;
}

// CHECK-LABEL: @float_int_overflow
// CHECK-TRAP-LABEL: @float_int_overflow
int float_int_overflow(float f) {
  // CHECK: %[[GE:.*]] = fcmp ogt float %[[F:.*]], 0xC1E0000020000000
  // CHECK: %[[LE:.*]] = fcmp olt float %[[F]], 0x41E0000000000000
  // CHECK: and i1 %[[GE]], %[[LE]]

  // CHECK: %[[CAST:.*]] = bitcast float %[[F]] to i32
  // CHECK: %[[ARG:.*]] = zext i32 %[[CAST]] to i64
  // CHECK: call void @__ubsan_handle_float_cast_overflow({{.*}}, i64 %[[ARG]]

  // CHECK-TRAP: %[[GE:.*]] = fcmp ogt float %[[F:.*]], 0xC1E0000020000000
  // CHECK-TRAP: %[[LE:.*]] = fcmp olt float %[[F]], 0x41E0000000000000
  // CHECK-TRAP: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  return f;
}

// CHECK-LABEL: @long_double_int_overflow
// CHECK-TRAP-LABEL: @long_double_int_overflow
int long_double_int_overflow(long double ld) {
  // CHECK: alloca x86_fp80
  // CHECK: %[[GE:.*]] = fcmp ogt x86_fp80 %[[F:.*]], 0xKC01E8000000100000000
  // CHECK: %[[LE:.*]] = fcmp olt x86_fp80 %[[F]], 0xK401E8000000000000000
  // CHECK: and i1 %[[GE]], %[[LE]]

  // CHECK: store x86_fp80 %[[F]], x86_fp80* %[[ALLOCA:.*]], !nosanitize
  // CHECK: %[[ARG:.*]] = ptrtoint x86_fp80* %[[ALLOCA]] to i64
  // CHECK: call void @__ubsan_handle_float_cast_overflow({{.*}}, i64 %[[ARG]]

  // CHECK-TRAP: %[[GE:.*]] = fcmp ogt x86_fp80 %[[F:.*]], 0xKC01E800000010000000
  // CHECK-TRAP: %[[LE:.*]] = fcmp olt x86_fp80 %[[F]], 0xK401E800000000000000
  // CHECK-TRAP: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  return ld;
}

// CHECK-LABEL: @float_uint_overflow
// CHECK-TRAP-LABEL: @float_uint_overflow
unsigned float_uint_overflow(float f) {
  // CHECK: %[[GE:.*]] = fcmp ogt float %[[F:.*]], -1.{{0*}}e+00
  // CHECK: %[[LE:.*]] = fcmp olt float %[[F]], 0x41F0000000000000
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(

  // CHECK-TRAP: %[[GE:.*]] = fcmp ogt float %[[F:.*]], -1.{{0*}}e+00
  // CHECK-TRAP: %[[LE:.*]] = fcmp olt float %[[F]], 0x41F0000000000000
  // CHECK-TRAP: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  return f;
}

// CHECK-LABEL: @fp16_char_overflow
// CHECK-TRAP-LABEL: @fp16_char_overflow
signed char fp16_char_overflow(__fp16 *p) {
  // CHECK: %[[GE:.*]] = fcmp ogt float %[[F:.*]], -1.29{{0*}}e+02
  // CHECK: %[[LE:.*]] = fcmp olt float %[[F]], 1.28{{0*}}e+02
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(

  // CHECK-TRAP: %[[GE:.*]] = fcmp ogt float %[[F:.*]], -1.29{{0*}}e+02
  // CHECK-TRAP: %[[LE:.*]] = fcmp olt float %[[F]], 1.28{{0*}}e+02
  // CHECK-TRAP: %[[INBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  return *p;
}

// CHECK-LABEL: @float_float_overflow
// CHECK-TRAP-LABEL: @float_float_overflow
float float_float_overflow(double f) {
  // CHECK: %[[F:.*]] = call double @llvm.fabs.f64(
  // CHECK: %[[GE:.*]] = fcmp ogt double %[[F]], 0x47EFFFFFE0000000
  // CHECK: %[[LE:.*]] = fcmp olt double %[[F]], 0x7FF0000000000000
  // CHECK: and i1 %[[GE]], %[[LE]]
  // CHECK: call void @__ubsan_handle_float_cast_overflow(

  // CHECK-TRAP: %[[F:.*]] = call double @llvm.fabs.f64(
  // CHECK-TRAP: %[[GE:.*]] = fcmp ogt double %[[F]], 0x47EFFFFFE0000000
  // CHECK-TRAP: %[[LE:.*]] = fcmp olt double %[[F]], 0x7FF0000000000000
  // CHECK-TRAP: %[[OUTOFBOUNDS:.*]] = and i1 %[[GE]], %[[LE]]
  // CHECK-TRAP: %[[INBOUNDS:.*]] = xor i1 %[[OUTOFBOUNDS]], true
  // CHECK-TRAP-NEXT: br i1 %[[INBOUNDS]]

  // CHECK-TRAP:      call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP-NEXT: unreachable
  return f;
}

// CHECK-LABEL:          @int_divide_overflow
// CHECK-OVERFLOW-LABEL: @int_divide_overflow
int int_divide_overflow(int a, int b) {
  // CHECK:               %[[ZERO:.*]] = icmp ne i32 %[[B:.*]], 0
  // CHECK-OVERFLOW-NOT:  icmp ne i32 %{{.*}}, 0
  // CHECK-TRAP:          %[[ZERO:.*]] = icmp ne i32 %[[B:.*]], 0

  // CHECK:               %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-NEXT:          %[[BOK:.*]] = icmp ne i32 %[[B]], -1
  // CHECK-NEXT:          %[[OVER:.*]] = or i1 %[[AOK]], %[[BOK]]

  // CHECK-OVERFLOW:      %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-OVERFLOW-NEXT: %[[BOK:.*]] = icmp ne i32 %[[B:.*]], -1
  // CHECK-OVERFLOW-NEXT: %[[OK:.*]] = or i1 %[[AOK]], %[[BOK]]

  // CHECK-TRAP:          %[[AOK:.*]] = icmp ne i32 %[[A:.*]], -2147483648
  // CHECK-TRAP-NEXT:     %[[BOK:.*]] = icmp ne i32 %[[B]], -1
  // CHECK-TRAP-NEXT:     %[[OVER:.*]] = or i1 %[[AOK]], %[[BOK]]

  // CHECK:               %[[OK:.*]] = and i1 %[[ZERO]], %[[OVER]]

  // CHECK:               br i1 %[[OK]]
  // CHECK-OVERFLOW:      br i1 %[[OK]]

  // CHECK-TRAP:          %[[OK:.*]] = and i1 %[[ZERO]], %[[OVER]]
  // CHECK-TRAP:          br i1 %[[OK]]

  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  return a / b;

  // CHECK:          }
  // CHECK-OVERFLOW: }
  // CHECK-TRAP:     }
}

// CHECK-LABEL: @sour_bool
_Bool sour_bool(_Bool *p) {
  // CHECK: %[[OK:.*]] = icmp ule i8 {{.*}}, 1
  // CHECK: br i1 %[[OK]]
  // CHECK: call void @__ubsan_handle_load_invalid_value(i8* bitcast ({{.*}}), i64 {{.*}})

  // CHECK-TRAP: %[[OK:.*]] = icmp ule i8 {{.*}}, 1
  // CHECK-TRAP: br i1 %[[OK]]

  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  return *p;
}

// CHECK-LABEL: @ret_nonnull
__attribute__((returns_nonnull))
int *ret_nonnull(int *a) {
  // CHECK: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK: br i1 [[OK]]
  // CHECK: call void @__ubsan_handle_nonnull_return

  // CHECK-TRAP: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK-TRAP: br i1 [[OK]]
  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  return a;
}

// CHECK-LABEL: @call_decl_nonnull
__attribute__((nonnull)) void decl_nonnull(int *a);
void call_decl_nonnull(int *a) {
  // CHECK: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK: br i1 [[OK]]
  // CHECK: call void @__ubsan_handle_nonnull_arg

  // CHECK-TRAP: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK-TRAP: br i1 [[OK]]
  // CHECK-TRAP: call void @llvm.trap() [[NR_NUW]]
  // CHECK-TRAP: unreachable
  decl_nonnull(a);
}

// CHECK-LABEL: @call_nonnull_variadic
__attribute__((nonnull)) void nonnull_variadic(int a, ...);
void call_nonnull_variadic(int a, int *b) {
  // CHECK: [[OK:%.*]] = icmp ne i32* {{.*}}, null
  // CHECK: br i1 [[OK]]
  // CHECK: call void @__ubsan_handle_nonnull_arg
  // CHECK-NOT: __ubsan_handle_nonnull_arg
  // CHECK: call void (i32, ...)* @nonnull_variadic
  nonnull_variadic(a, b);
}

// CHECK: ![[WEIGHT_MD]] = metadata !{metadata !"branch_weights", i32 1048575, i32 1}

// CHECK-TRAP: attributes [[NR_NUW]] = { noreturn nounwind }
