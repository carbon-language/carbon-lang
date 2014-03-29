; RUN: llc < %s -march=arm64 -arm64-neon-syntax=apple -arm64-simd-scalar=true -asm-verbose=false | FileCheck %s
;
define <2 x i64> @bar(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: bar:
; CHECK: add.2d	v[[REG:[0-9]+]], v0, v1
; CHECK: add	d[[REG3:[0-9]+]], d[[REG]], d1
; CHECK: sub	d[[REG2:[0-9]+]], d[[REG]], d1
  %add = add <2 x i64> %a, %b
  %vgetq_lane = extractelement <2 x i64> %add, i32 0
  %vgetq_lane2 = extractelement <2 x i64> %b, i32 0
  %add3 = add i64 %vgetq_lane, %vgetq_lane2
  %sub = sub i64 %vgetq_lane, %vgetq_lane2
  %vecinit = insertelement <2 x i64> undef, i64 %add3, i32 0
  %vecinit8 = insertelement <2 x i64> %vecinit, i64 %sub, i32 1
  ret <2 x i64> %vecinit8
}

define double @subdd_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: subdd_su64:
; CHECK: sub d0, d1, d0
; CHECK-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %sub.i = sub nsw i64 %vecext1, %vecext
  %retval = bitcast i64 %sub.i to double
  ret double %retval
}

define double @vaddd_su64(<2 x i64> %a, <2 x i64> %b) nounwind readnone {
; CHECK-LABEL: vaddd_su64:
; CHECK: add d0, d1, d0
; CHECK-NEXT: ret
  %vecext = extractelement <2 x i64> %a, i32 0
  %vecext1 = extractelement <2 x i64> %b, i32 0
  %add.i = add nsw i64 %vecext1, %vecext
  %retval = bitcast i64 %add.i to double
  ret double %retval
}
