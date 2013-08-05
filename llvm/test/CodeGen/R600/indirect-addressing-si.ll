; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

; Tests for indirect addressing on SI, which is implemented using dynamic
; indexing of vectors.

; CHECK: extract_w_offset
; CHECK: S_MOV_B32 M0
; CHECK-NEXT: V_MOVRELS_B32_e32
define void @extract_w_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 %in, 1
  %1 = extractelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 %0
  store float %1, float addrspace(1)* %out
  ret void
}

; CHECK: extract_wo_offset
; CHECK: S_MOV_B32 M0
; CHECK-NEXT: V_MOVRELS_B32_e32
define void @extract_wo_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = extractelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 %in
  store float %0, float addrspace(1)* %out
  ret void
}

; CHECK: insert_w_offset
; CHECK: S_MOV_B32 M0
; CHECK-NEXT: V_MOVRELD_B32_e32
define void @insert_w_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 %in, 1
  %1 = insertelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, float 5.0, i32 %0
  %2 = extractelement <4 x float> %1, i32 2
  store float %2, float addrspace(1)* %out
  ret void
}

; CHECK: insert_wo_offset
; CHECK: S_MOV_B32 M0
; CHECK-NEXT: V_MOVRELD_B32_e32
define void @insert_wo_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = insertelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, float 5.0, i32 %in
  %1 = extractelement <4 x float> %0, i32 2
  store float %1, float addrspace(1)* %out
  ret void
}
