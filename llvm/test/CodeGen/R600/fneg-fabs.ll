; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s


; DAGCombiner will transform:
; (fabs (f32 bitcast (i32 a))) => (f32 bitcast (and (i32 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; FUNC-LABEL: @fneg_fabs_free_f32
; R600-NOT: AND
; R600: |PV.{{[XYZW]}}|
; R600: -PV

; SI: V_OR_B32
define void @fneg_fabs_free_f32(float addrspace(1)* %out, i32 %in) {
  %bc = bitcast i32 %in to float
  %fabs = call float @llvm.fabs.f32(float %bc)
  %fsub = fsub float -0.000000e+00, %fabs
  store float %fsub, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fneg_fabs_fn_free_f32
; R600-NOT: AND
; R600: |PV.{{[XYZW]}}|
; R600: -PV

; SI: V_OR_B32
define void @fneg_fabs_fn_free_f32(float addrspace(1)* %out, i32 %in) {
  %bc = bitcast i32 %in to float
  %fabs = call float @fabs(float %bc)
  %fsub = fsub float -0.000000e+00, %fabs
  store float %fsub, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fneg_fabs_v2f32
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: -PV
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: -PV

; SI: V_OR_B32
; SI: V_OR_B32
define void @fneg_fabs_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %in)
  %fsub = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %fabs
  store <2 x float> %fsub, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fneg_fabs_v4f32
; SI: V_OR_B32
; SI: V_OR_B32
; SI: V_OR_B32
; SI: V_OR_B32
define void @fneg_fabs_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
  %fabs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %in)
  %fsub = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %fabs
  store <4 x float> %fsub, <4 x float> addrspace(1)* %out
  ret void
}

declare float @fabs(float) readnone
declare float @llvm.fabs.f32(float) readnone
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) readnone
declare <4 x float> @llvm.fabs.v4f32(<4 x float>) readnone
