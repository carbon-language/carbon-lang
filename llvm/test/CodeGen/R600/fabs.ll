; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s


; DAGCombiner will transform:
; (fabs (f32 bitcast (i32 a))) => (f32 bitcast (and (i32 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; FUNC-LABEL: {{^}}fabs_fn_free:
; R600-NOT: AND
; R600: |PV.{{[XYZW]}}|

; SI: V_AND_B32

define void @fabs_fn_free(float addrspace(1)* %out, i32 %in) {
  %bc= bitcast i32 %in to float
  %fabs = call float @fabs(float %bc)
  store float %fabs, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_free:
; R600-NOT: AND
; R600: |PV.{{[XYZW]}}|

; SI: V_AND_B32

define void @fabs_free(float addrspace(1)* %out, i32 %in) {
  %bc= bitcast i32 %in to float
  %fabs = call float @llvm.fabs.f32(float %bc)
  store float %fabs, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_f32:
; R600: |{{(PV|T[0-9])\.[XYZW]}}|

; SI: V_AND_B32
define void @fabs_f32(float addrspace(1)* %out, float %in) {
  %fabs = call float @llvm.fabs.f32(float %in)
  store float %fabs, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_v2f32:
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|

; SI: V_AND_B32
; SI: V_AND_B32
define void @fabs_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %in)
  store <2 x float> %fabs, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_v4f32:
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|

; SI: V_AND_B32
; SI: V_AND_B32
; SI: V_AND_B32
; SI: V_AND_B32
define void @fabs_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
  %fabs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %in)
  store <4 x float> %fabs, <4 x float> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fabs_fn_fold:
; SI: S_LOAD_DWORD [[ABS_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; SI-NOT: AND
; SI: V_MUL_F32_e64 v{{[0-9]+}}, |[[ABS_VALUE]]|, v{{[0-9]+}}
define void @fabs_fn_fold(float addrspace(1)* %out, float %in0, float %in1) {
  %fabs = call float @fabs(float %in0)
  %fmul = fmul float %fabs, %in1
  store float %fmul, float addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fabs_fold:
; SI: S_LOAD_DWORD [[ABS_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; SI-NOT: AND
; SI: V_MUL_F32_e64 v{{[0-9]+}}, |[[ABS_VALUE]]|, v{{[0-9]+}}
define void @fabs_fold(float addrspace(1)* %out, float %in0, float %in1) {
  %fabs = call float @llvm.fabs.f32(float %in0)
  %fmul = fmul float %fabs, %in1
  store float %fmul, float addrspace(1)* %out
  ret void
}

declare float @fabs(float) readnone
declare float @llvm.fabs.f32(float) readnone
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) readnone
declare <4 x float> @llvm.fabs.v4f32(<4 x float>) readnone
