; RUN: not llvm-as -opaque-pointers < %s -o /dev/null 2>&1 | FileCheck %s

define <4 x float> @transpose(<4 x float> %m, i32 %arg) {
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.3 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.2, i32 %arg, i32 2)
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.4 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.3, i32 2, i32 %arg)
  %result.0 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %m, i32 0, i32 0)
  %result.1 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.0, i32 3, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.1, i32 2, i32 1)
  %result.3 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.2, i32 %arg, i32 2)
  %result.4 = call <4 x float> @llvm.matrix.transpose.v4f32(<4 x float> %result.3, i32 2, i32 %arg)
  ret <4 x float> %result.4
}

define <4 x float> @multiply(<4 x float> %m, i32 %arg) {
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.3 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.2, <4 x float> %m, i32 %arg, i32 2, i32 1)
  %result.0 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %m, <4 x float> %m, i32 0, i32 0, i32 0)
  %result.1 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.0, <4 x float> %m, i32 3, i32 2, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.1, <4 x float> %m, i32 2, i32 2, i32 1)
  %result.3 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %result.2, <4 x float> %m, i32 %arg, i32 2, i32 1)
  ret <4 x float> %result.3
}

define <4 x float> @column.major_load(ptr %m, ptr %n, i32 %arg) {
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %arg
; CHECK-NEXT:   %result.3 = call <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr %n, i64 2, i1 true, i32 3, i32 %arg)
  %result.0 = call <4 x float> @llvm.matrix.column.major.load.v4f32.i64(ptr %m, i64 0, i1 false, i32 0, i32 0)
  %result.1 = call <4 x float> @llvm.matrix.column.major.load.v4f32.i64(ptr %m, i64 2, i1 false, i32 1, i32 2)
  %result.2 = call <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr %n, i64 2, i1 true, i32 3, i32 3)
  %result.3 = call <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr %n, i64 2, i1 true, i32 3, i32 %arg)
  ret <4 x float> %result.1
}

define void @column.major_store(ptr %m, ptr %n, i64 %arg) {
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
; CHECK-NEXT: Result of a matrix operation does not fit in the returned vector!
  call void @llvm.matrix.column.major.store.v4f32.i64(<4 x float> zeroinitializer, ptr %m, i64 0, i1 false, i32 0, i32 0)
  call void @llvm.matrix.column.major.store.v4f32.i64(<4 x float> zeroinitializer, ptr %m, i64 2, i1 false, i32 1, i32 2)
  call void @llvm.matrix.column.major.store.v6f32.i64(<6 x float> zeroinitializer, ptr %n, i64 2, i1 false, i32 3, i32 3)
  call void @llvm.matrix.column.major.store.v6f32.i64(<6 x float> zeroinitializer, ptr %n, i64 %arg, i1 false, i32 3, i32 3)
  ret void
}

define <4 x float> @transpose_mixed_types(<4 x float> %fvec, <4 x i32> %ivec, i32 %arg) {
;
; CHECK-NEXT: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.matrix.transpose.v4f32.v4i32
; CHECK-NEXT: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.matrix.transpose.v4i32.v4f32
;
  %result.0 = call <4 x float> @llvm.matrix.transpose.v4f32.v4i32(<4 x i32> %ivec, i32 0, i32 0)
  %result.1 = call <4 x i32> @llvm.matrix.transpose.v4i32.v4f32(<4 x float> %result.0, i32 3, i32 2)
  ret <4 x float> %result.0
}

define <4 x float> @multiply_mixed_types(<4 x i32> %ivec, <4 x float> %fvec, i32 %arg) {
;
; CHECK-NEXT: Vector element type mismatch of the result and first operand vector!
; CHECK-NEXT: ptr @llvm.matrix.multiply.v4i32.v4f32.v4f32
; CHECK-NEXT: Vector element type mismatch of the result and first operand vector!
; CHECK-NEXT: ptr @llvm.matrix.multiply.v4f32.v4i32.v4f32
; CHECK-NEXT: Vector element type mismatch of the result and second operand vector!
; CHECK-NEXT: ptr @llvm.matrix.multiply.v4f32.v4f32.v4i32
; CHECK-NEXT: Vector element type mismatch of the result and first operand vector!
; CHECK-NEXT: ptr @llvm.matrix.multiply.v4f32.v4i32.v4i32
;
  %result.0 = call <4 x i32> @llvm.matrix.multiply.v4i32.v4f32.v4f32(<4 x float> %fvec, <4 x float> %fvec, i32 2, i32 2, i32 2)
  %result.1 = call <4 x float> @llvm.matrix.multiply.v4f32.v4i32.v4f32(<4 x i32> %result.0, <4 x float> %fvec, i32 2, i32 2, i32 2)
  %result.2 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4i32(<4 x float> %fvec, <4 x i32> %ivec, i32 2, i32 2, i32 2)
  %result.3 = call <4 x float> @llvm.matrix.multiply.v4f32.v4i32.v4i32(<4 x i32> %ivec, <4 x i32> %ivec, i32 2, i32 2, i32 2)
  ret <4 x float> %result.3
}

define void @column.major_store_non_int_float_type(ptr %m, ptr %n, i64 %arg) {
;
; CHECK-NEXT: Result type must be an integer or floating-point type!
; CHECK-NEXT: ptr @llvm.matrix.column.major.store.v4p0.i64
;
  call void @llvm.matrix.column.major.store.v4p0.i64(<4 x ptr> zeroinitializer, ptr %n, i64 2, i1 false, i32 2, i32 2)
  ret void
}

define <4 x float> @column.major_load_stride_too_small(ptr %m, i32 %arg) {
;
; CHECK-NEXT: Stride must be greater or equal than the number of rows!
; CHECK-NEXT: ptr @llvm.matrix.column.major.load.v4f32.i64
;
  %result.1 = call <4 x float> @llvm.matrix.column.major.load.v4f32.i64(ptr %m, i64 1, i1 false, i32 2, i32 2)
  ret <4 x float> %result.1
}

define void @column.major_store_stride_too_small(ptr %m, i64 %arg) {
;
; CHECK-NEXT: Stride must be greater or equal than the number of rows!
; CHECK-NEXT: ptr @llvm.matrix.column.major.store.v4f32.i64
;
  call void @llvm.matrix.column.major.store.v4f32.i64(<4 x float> zeroinitializer, ptr %m, i64 1, i1 false, i32 2, i32 2)
  ret void
}

declare <4 x i32>   @llvm.matrix.column.major.load.v4i32.i64(ptr, i64, i1, i32, i32)
declare <4 x float> @llvm.matrix.column.major.load.v4f32.p0(ptr, i64, i1, i32, i32)
declare <4 x float> @llvm.matrix.column.major.load.v4f32.i64(ptr, i64, i1, i32, i32)
declare <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr, i64, i1, i32, i32)

declare void @llvm.matrix.column.major.store.v4f32.i64(<4 x float>, ptr, i64, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v6f32.i64(<6 x float>, ptr, i64, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v4i32.vi32(<4 x i32>, ptr, i64, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v4f32.p0(<4 x float>, ptr, i64, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v4p0.i64(<4 x ptr>, ptr, i64, i1, i32, i32)

declare <4 x i32>   @llvm.matrix.transpose.v4i32.v4f32(<4 x float>, i32, i32)
declare <4 x float> @llvm.matrix.transpose.v4f32(<4 x float>, i32, i32)
declare <4 x float> @llvm.matrix.transpose.v4f32.v4i32(<4 x i32>, i32, i32)

declare <4 x i32>   @llvm.matrix.multiply.v4i32.v4f32.v4f32(<4 x float>, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.matrix.multiply.v4f32.v4i32.v4f32(<4 x i32>, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4i32(<4 x float>, <4 x i32>, i32, i32, i32)
declare <4 x float> @llvm.matrix.multiply.v4f32.v4i32.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32)
declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float>, <4 x float>, i32, i32, i32)
