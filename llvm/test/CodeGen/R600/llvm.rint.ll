; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck %s -check-prefix=R600 -check-prefix=FUNC
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: @f32
; R600: RNDNE

; SI: V_RNDNE_F32_e32
define void @f32(float addrspace(1)* %out, float %in) {
entry:
  %0 = call float @llvm.rint.f32(float %in)
  store float %0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @v2f32
; R600: RNDNE
; R600: RNDNE

; SI: V_RNDNE_F32_e32
; SI: V_RNDNE_F32_e32
define void @v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.rint.v2f32(<2 x float> %in)
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @v4f32
; R600: RNDNE
; R600: RNDNE
; R600: RNDNE
; R600: RNDNE

; SI: V_RNDNE_F32_e32
; SI: V_RNDNE_F32_e32
; SI: V_RNDNE_F32_e32
; SI: V_RNDNE_F32_e32
define void @v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.rint.v4f32(<4 x float> %in)
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

declare float @llvm.rint.f32(float) #0
declare <2 x float> @llvm.rint.v2f32(<2 x float>) #0
declare <4 x float> @llvm.rint.v4f32(<4 x float>) #0

attributes #0 = { nounwind readonly }
