; RUN: opt < %s -instcombine -S | FileCheck %s

; If the mask isn't constant, do nothing.

define void @mstore(i8* %f, <4 x i32> %mask, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> %mask, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore(
; CHECK-NEXT:  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> %mask, <4 x float> %v)
; CHECK-NEXT:  ret void
}

; Zero mask is a nop.

define void @mstore_zeros(i8* %f, <4 x float> %v)  {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> zeroinitializer, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_zeros(
; CHECK-NEXT:  ret void
}

; Only the sign bit matters.

define void @mstore_fake_ones(i8* %f, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> <i32 1, i32 2, i32 3, i32 2147483647>, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_fake_ones(
; CHECK-NEXT:  ret void
}

; All mask bits are set, so this is just a vector store.

define void @mstore_real_ones(i8* %f, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> <i32 -1, i32 -2, i32 -3, i32 -2147483648>, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_real_ones(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x float>*
; CHECK-NEXT:  store <4 x float> %v, <4 x float>* %castvec
; CHECK-NEXT:  ret void
}

; It's a constant mask, so convert to an LLVM intrinsic. The backend should optimize further.

define void @mstore_one_one(i8* %f, <4 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps(i8* %f, <4 x i32> <i32 0, i32 0, i32 0, i32 -1>, <4 x float> %v)
  ret void

; CHECK-LABEL: @mstore_one_one(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x float>*
; CHECK-NEXT:  call void @llvm.masked.store.v4f32(<4 x float> %v, <4 x float>* %castvec, i32 1, <4 x i1> <i1 false, i1 false, i1 false, i1 true>)
; CHECK-NEXT:  ret void
}

; Try doubles.

define void @mstore_one_one_double(i8* %f, <2 x double> %v) {
  tail call void @llvm.x86.avx.maskstore.pd(i8* %f, <2 x i64> <i64 -1, i64 0>, <2 x double> %v)
  ret void

; CHECK-LABEL: @mstore_one_one_double(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <2 x double>*
; CHECK-NEXT:  call void @llvm.masked.store.v2f64(<2 x double> %v, <2 x double>* %castvec, i32 1, <2 x i1> <i1 true, i1 false>)
; CHECK-NEXT:  ret void
}

define void @mstore_v8f32(i8* %f, <8 x float> %v) {
  tail call void @llvm.x86.avx.maskstore.ps.256(i8* %f, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 -1, i32 -2, i32 -3, i32 -4>, <8 x float> %v)
  ret void

; CHECK-LABEL: @mstore_v8f32(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <8 x float>*
; CHECK-NEXT:  call void @llvm.masked.store.v8f32(<8 x float> %v, <8 x float>* %castvec, i32 1, <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:  ret void
}

define void @mstore_v4f64(i8* %f, <4 x double> %v) {
  tail call void @llvm.x86.avx.maskstore.pd.256(i8* %f, <4 x i64> <i64 -1, i64 0, i64 1, i64 2>, <4 x double> %v)
  ret void

; CHECK-LABEL: @mstore_v4f64(
; CHECK-NEXT:  %castvec = bitcast i8* %f to <4 x double>*
; CHECK-NEXT:  call void @llvm.masked.store.v4f64(<4 x double> %v, <4 x double>* %castvec, i32 1, <4 x i1> <i1 true, i1 false, i1 false, i1 false>)
; CHECK-NEXT:  ret void
}

declare void @llvm.x86.avx.maskstore.ps(i8*, <4 x i32>, <4 x float>)
declare void @llvm.x86.avx.maskstore.pd(i8*, <2 x i64>, <2 x double>)
declare void @llvm.x86.avx.maskstore.ps.256(i8*, <8 x i32>, <8 x float>)
declare void @llvm.x86.avx.maskstore.pd.256(i8*, <4 x i64>, <4 x double>)

