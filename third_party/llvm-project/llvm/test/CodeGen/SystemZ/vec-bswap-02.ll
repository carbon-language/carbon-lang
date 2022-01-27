; Test stores of byte-swapped vector elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Test v8i16 stores.
define void @f1(<8 x i16> %val, <8 x i16> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vstbrh %v24, 0(%r2)
; CHECK: br %r14
  %swap = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %val)
  store <8 x i16> %swap, <8 x i16> *%ptr
  ret void
}

; Test v4i32 stores.
define void @f2(<4 x i32> %val, <4 x i32> *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vstbrf %v24, 0(%r2)
; CHECK: br %r14
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Test v2i64 stores.
define void @f3(<2 x i64> %val, <2 x i64> *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vstbrg %v24, 0(%r2)
; CHECK: br %r14
  %swap = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %val)
  store <2 x i64> %swap, <2 x i64> *%ptr
  ret void
}

; Test the highest aligned in-range offset.
define void @f4(<4 x i32> %val, <4 x i32> *%base) {
; CHECK-LABEL: f4:
; CHECK: vstbrf %v24, 4080(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 255
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Test the highest unaligned in-range offset.
define void @f5(<4 x i32> %val, i8 *%base) {
; CHECK-LABEL: f5:
; CHECK: vstbrf %v24, 4095(%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 4095
  %ptr = bitcast i8 *%addr to <4 x i32> *
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  store <4 x i32> %swap, <4 x i32> *%ptr, align 1
  ret void
}

; Test the next offset up, which requires separate address logic,
define void @f6(<4 x i32> %val, <4 x i32> *%base) {
; CHECK-LABEL: f6:
; CHECK: aghi %r2, 4096
; CHECK: vstbrf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 256
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Test negative offsets, which also require separate address logic,
define void @f7(<4 x i32> %val, <4 x i32> *%base) {
; CHECK-LABEL: f7:
; CHECK: aghi %r2, -16
; CHECK: vstbrf %v24, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr <4 x i32>, <4 x i32> *%base, i64 -1
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  store <4 x i32> %swap, <4 x i32> *%ptr
  ret void
}

; Check that indexes are allowed.
define void @f8(<4 x i32> %val, i8 *%base, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: vstbrf %v24, 0(%r3,%r2)
; CHECK: br %r14
  %addr = getelementptr i8, i8 *%base, i64 %index
  %ptr = bitcast i8 *%addr to <4 x i32> *
  %swap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val)
  store <4 x i32> %swap, <4 x i32> *%ptr, align 1
  ret void
}

