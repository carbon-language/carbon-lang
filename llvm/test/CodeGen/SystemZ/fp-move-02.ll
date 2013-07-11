; Test moves between FPRs and GPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test 32-bit moves from GPRs to FPRs.  The GPR must be moved into the high
; 32 bits of the FPR.
define float @f1(i32 %a) {
; CHECK: f1:
; CHECK: sllg [[REGISTER:%r[0-5]]], %r2, 32
; CHECK: ldgr %f0, [[REGISTER]]
  %res = bitcast i32 %a to float
  ret float %res
}

; Like f1, but create a situation where the shift can be folded with
; surrounding code.
define float @f2(i64 %big) {
; CHECK: f2:
; CHECK: sllg [[REGISTER:%r[0-5]]], %r2, 31
; CHECK: ldgr %f0, [[REGISTER]]
  %shift = lshr i64 %big, 1
  %a = trunc i64 %shift to i32
  %res = bitcast i32 %a to float
  ret float %res
}

; Another example of the same thing.
define float @f3(i64 %big) {
; CHECK: f3:
; CHECK: sllg [[REGISTER:%r[0-5]]], %r2, 2
; CHECK: ldgr %f0, [[REGISTER]]
  %shift = ashr i64 %big, 30
  %a = trunc i64 %shift to i32
  %res = bitcast i32 %a to float
  ret float %res
}

; Like f1, but the value to transfer is already in the high 32 bits.
define float @f4(i64 %big) {
; CHECK: f4:
; CHECK-NOT: %r2
; CHECK: risbg [[REG:%r[0-5]]], %r2, 0, 159, 0
; CHECK-NOT: [[REG]]
; CHECK: ldgr %f0, [[REG]]
  %shift = ashr i64 %big, 32
  %a = trunc i64 %shift to i32
  %res = bitcast i32 %a to float
  ret float %res
}

; Test 64-bit moves from GPRs to FPRs.
define double @f5(i64 %a) {
; CHECK: f5:
; CHECK: ldgr %f0, %r2
  %res = bitcast i64 %a to double
  ret double %res
}

; Test 128-bit moves from GPRs to FPRs.  i128 isn't a legitimate type,
; so this goes through memory.
; FIXME: it would be better to use one MVC here.
define void @f6(fp128 *%a, i128 *%b) {
; CHECK: f6:
; CHECK: lg
; CHECK: mvc
; CHECK: stg
; CHECK: br %r14
  %val = load i128 *%b
  %res = bitcast i128 %val to fp128
  store fp128 %res, fp128 *%a
  ret void
}

; Test 32-bit moves from FPRs to GPRs.  The high 32 bits of the FPR should
; be moved into the low 32 bits of the GPR.
define i32 @f7(float %a) {
; CHECK: f7:
; CHECK: lgdr [[REGISTER:%r[0-5]]], %f0
; CHECK: srlg %r2, [[REGISTER]], 32
  %res = bitcast float %a to i32
  ret i32 %res
}

; Test 64-bit moves from FPRs to GPRs.
define i64 @f8(double %a) {
; CHECK: f8:
; CHECK: lgdr %r2, %f0
  %res = bitcast double %a to i64
  ret i64 %res
}

; Test 128-bit moves from FPRs to GPRs, with the same restriction as f6.
define void @f9(fp128 *%a, i128 *%b) {
; CHECK: f9:
; CHECK: ld
; CHECK: ld
; CHECK: std
; CHECK: std
  %val = load fp128 *%a
  %res = bitcast fp128 %val to i128
  store i128 %res, i128 *%b
  ret void
}

