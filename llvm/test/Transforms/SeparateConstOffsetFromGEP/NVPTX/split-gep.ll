; RUN: opt < %s -separate-const-offset-from-gep -dce -S | FileCheck %s

; Several unit tests for -separate-const-offset-from-gep. The transformation
; heavily relies on TargetTransformInfo, so we put these tests under
; target-specific folders.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
; target triple is necessary; otherwise TargetTransformInfo rejects any
; addressing mode.
target triple = "nvptx64-unknown-unknown"

%struct.S = type { float, double }

@struct_array = global [1024 x %struct.S] zeroinitializer, align 16
@float_2d_array = global [32 x [32 x float]] zeroinitializer, align 4

; We should not extract any struct field indices, because fields in a struct
; may have different types.
define double* @struct(i32 %i) {
entry:
  %add = add nsw i32 %i, 5
  %idxprom = sext i32 %add to i64
  %p = getelementptr inbounds [1024 x %struct.S]* @struct_array, i64 0, i64 %idxprom, i32 1
  ret double* %p
}
; CHECK-LABEL: @struct
; CHECK: getelementptr [1024 x %struct.S]* @struct_array, i64 0, i32 %i, i32 1

; We should be able to trace into sext/zext if it's directly used as a GEP
; index.
define float* @sext_zext(i32 %i, i32 %j) {
entry:
  %i1 = add i32 %i, 1
  %j2 = add i32 %j, 2
  %i1.ext = sext i32 %i1 to i64
  %j2.ext = zext i32 %j2 to i64
  %p = getelementptr inbounds [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i1.ext, i64 %j2.ext
  ret float* %p
}
; CHECK-LABEL: @sext_zext
; CHECK: getelementptr [32 x [32 x float]]* @float_2d_array, i64 0, i32 %i, i32 %j
; CHECK: add i64 %{{[0-9]+}}, 136

; We should be able to trace into sext/zext if it can be distributed to both
; operands, e.g., sext (add nsw a, b) == add nsw (sext a), (sext b)
define float* @ext_add_no_overflow(i64 %a, i32 %b, i64 %c, i32 %d) {
  %b1 = add nsw i32 %b, 1
  %b2 = sext i32 %b1 to i64
  %i = add i64 %a, %b2
  %d1 = add nuw i32 %d, 1
  %d2 = zext i32 %d1 to i64
  %j = add i64 %c, %d2
  %p = getelementptr inbounds [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 %j
  ret float* %p
}
; CHECK-LABEL: @ext_add_no_overflow
; CHECK: [[BASE_PTR:%[0-9]+]] = getelementptr [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}
; CHECK: [[BASE_INT:%[0-9]+]] = ptrtoint float* [[BASE_PTR]] to i64
; CHECK: add i64 [[BASE_INT]], 132

; We should treat "or" with no common bits (%k) as "add", and leave "or" with
; potentially common bits (%l) as is.
define float* @or(i64 %i) {
entry:
  %j = shl i64 %i, 2
  %k = or i64 %j, 3 ; no common bits
  %l = or i64 %j, 4 ; potentially common bits
  %p = getelementptr inbounds [32 x [32 x float]]* @float_2d_array, i64 0, i64 %k, i64 %l
  ret float* %p
}
; CHECK-LABEL: @or
; CHECK: getelementptr [32 x [32 x float]]* @float_2d_array, i64 0, i64 %j, i64 %l
; CHECK: add i64 %{{[0-9]+}}, 384

; The subexpression (b + 5) is used in both "i = a + (b + 5)" and "*out = b +
; 5". When extracting the constant offset 5, make sure "*out = b + 5" isn't
; affected.
define float* @expr(i64 %a, i64 %b, i64* %out) {
entry:
  %b5 = add i64 %b, 5
  %i = add i64 %b5, %a
  %p = getelementptr inbounds [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 0
  store i64 %b5, i64* %out
  ret float* %p
}
; CHECK-LABEL: @expr
; CHECK: getelementptr [32 x [32 x float]]* @float_2d_array, i64 0, i64 %0, i64 0
; CHECK: add i64 %{{[0-9]+}}, 640
; CHECK: store i64 %b5, i64* %out

; Verifies we handle "sub" correctly.
define float* @sub(i64 %i, i64 %j) {
  %i2 = sub i64 %i, 5 ; i - 5
  %j2 = sub i64 5, %j ; 5 - i
  %p = getelementptr inbounds [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i2, i64 %j2
  ret float* %p
}
; CHECK-LABEL: @sub
; CHECK: %[[j2:[0-9]+]] = sub i64 0, %j
; CHECK: getelementptr [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 %[[j2]]
; CHECK: add i64 %{{[0-9]+}}, -620
