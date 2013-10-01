; Test moves between FPRs and GPRs for z196 and above.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check that moves from i32s to floats can use high registers.
define float @f1(i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: llhh [[REG:%r[0-5]]], 0(%r2)
; CHECK: oihh [[REG]], 16256
; CHECK: ldgr %f0, [[REG]]
; CHECK: br %r14
  %base = load i16 *%ptr
  %ext = zext i16 %base to i32
  %full = or i32 %ext, 1065353216
  %res = bitcast i32 %full to float
  ret float %res
}

; Check that moves from floats to i32s can use high registers.
; This "store the low byte" technique is used by llvmpipe, for example.
define void @f2(float %val, i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: lgdr [[REG:%r[0-5]]], %f0
; CHECK: stch [[REG]], 0(%r2)
; CHECK: br %r14
  %res = bitcast float %val to i32
  %trunc = trunc i32 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}
