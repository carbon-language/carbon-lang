;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test checks a bug in R600TargetLowering::LowerSELECT_CC where the
; chance to optimize the fcmp + select instructions to CNDE was missed
; due to the fact that the operands to fcmp and select had different types

;CHECK: CNDE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], literal.x, 0.0, -1}}

define void @test(i32 addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %0 = load float addrspace(1)* %in
  %cmp = fcmp oeq float %0, 0.000000e+00
  %value = select i1 %cmp, i32 -1, i32 0
  store i32 %value, i32 addrspace(1)* %out
  ret void
}
