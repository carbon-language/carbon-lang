; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: @fadd_f32
; R600-CHECK: ADD {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, KC0[2].W
; SI-CHECK: @fadd_f32
; SI-CHECK: V_ADD_F32
define void @fadd_f32(float addrspace(1)* %out, float %a, float %b) {
entry:
   %0 = fadd float %a, %b
   store float %0, float addrspace(1)* %out
   ret void
}

; R600-CHECK: @fadd_v2f32
; R600-CHECK-DAG: ADD {{\** *}}T{{[0-9]\.[XYZW]}}, KC0[3].X, KC0[3].Z
; R600-CHECK-DAG: ADD {{\** *}}T{{[0-9]\.[XYZW]}}, KC0[2].W, KC0[3].Y
; SI-CHECK: @fadd_v2f32
; SI-CHECK: V_ADD_F32
; SI-CHECK: V_ADD_F32
define void @fadd_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fadd <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; R600-CHECK: @fadd_v4f32
; R600-CHECK: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: ADD {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; SI-CHECK: @fadd_v4f32
; SI-CHECK: V_ADD_F32
; SI-CHECK: V_ADD_F32
; SI-CHECK: V_ADD_F32
; SI-CHECK: V_ADD_F32
define void @fadd_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float> addrspace(1) * %in
  %b = load <4 x float> addrspace(1) * %b_ptr
  %result = fadd <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
