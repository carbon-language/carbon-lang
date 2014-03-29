; RUN: llc < %s -march=arm64 -arm64-neon-syntax=apple | FileCheck %s

define i32 @cnt32_advsimd(i32 %x) nounwind readnone {
  %cnt = tail call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %cnt
; CHECK: fmov	s0, w0
; CHECK: cnt.8b	v0, v0
; CHECK: uaddlv.8b	h0, v0
; CHECK: fmov w0, s0
; CHECK: ret
}

define i64 @cnt64_advsimd(i64 %x) nounwind readnone {
  %cnt = tail call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %cnt
; CHECK: fmov	d0, x0
; CHECK: cnt.8b	v0, v0
; CHECK: uaddlv.8b	h0, v0
; CHECK: fmov	w0, s0
; CHECK: ret
}

; Do not use AdvSIMD when -mno-implicit-float is specified.
; rdar://9473858

define i32 @cnt32(i32 %x) nounwind readnone noimplicitfloat {
  %cnt = tail call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %cnt
; CHECK-LABEL: cnt32:
; CHECK-NOT 16b
; CHECK: ret
}

define i64 @cnt64(i64 %x) nounwind readnone noimplicitfloat {
  %cnt = tail call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %cnt
; CHECK-LABEL: cnt64:
; CHECK-NOT 16b
; CHECK: ret
}

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare i64 @llvm.ctpop.i64(i64) nounwind readnone
