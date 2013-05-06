; Test extensions of f32 to f128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register extension.
define void @f1(fp128 *%dst, float %val) {
; CHECK: f1:
; CHECK: lxebr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %res = fpext float %val to fp128
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the low end of the LXEB range.
define void @f2(fp128 *%dst, float *%ptr) {
; CHECK: f2:
; CHECK: lxeb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %val = load float *%ptr
  %res = fpext float %val to fp128
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the high end of the aligned LXEB range.
define void @f3(fp128 *%dst, float *%base) {
; CHECK: f3:
; CHECK: lxeb %f0, 4092(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1023
  %val = load float *%ptr
  %res = fpext float %val to fp128
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(fp128 *%dst, float *%base) {
; CHECK: f4:
; CHECK: aghi %r3, 4096
; CHECK: lxeb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1024
  %val = load float *%ptr
  %res = fpext float %val to fp128
  store fp128 %res, fp128 *%dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(fp128 *%dst, float *%base) {
; CHECK: f5:
; CHECK: aghi %r3, -4
; CHECK: lxeb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 -1
  %val = load float *%ptr
  %res = fpext float %val to fp128
  store fp128 %res, fp128 *%dst
  ret void
}

; Check that LXEB allows indices.
define void @f6(fp128 *%dst, float *%base, i64 %index) {
; CHECK: f6:
; CHECK: sllg %r1, %r4, 2
; CHECK: lxeb %f0, 400(%r1,%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float *%base, i64 %index
  %ptr2 = getelementptr float *%ptr1, i64 100
  %val = load float *%ptr2
  %res = fpext float %val to fp128
  store fp128 %res, fp128 *%dst
  ret void
}
