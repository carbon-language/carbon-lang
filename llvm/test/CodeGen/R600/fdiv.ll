; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; These tests check that fdiv is expanded correctly and also test that the
; scheduler is scheduling the RECIP_IEEE and MUL_IEEE instructions in separate
; instruction groups.

; CHECK: @fdiv_v2f32
; CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; CHECK-DAG: MUL_IEEE T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; CHECK-DAG: MUL_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS
define void @fdiv_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fdiv <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; CHECK: @fdiv_v4f32
; CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK-DAG: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; CHECK-DAG: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; CHECK-DAG: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; CHECK-DAG: MUL_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS

define void @fdiv_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float> addrspace(1) * %in
  %b = load <4 x float> addrspace(1) * %b_ptr
  %result = fdiv <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
