; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: @uint_to_fp_v2i32
; R600-DAG: UINT_TO_FLT * T{{[0-9]+\.[XYZW]}}, KC0[2].W
; R600-DAG: UINT_TO_FLT * T{{[0-9]+\.[XYZW]}}, KC0[3].X

; SI: V_CVT_F32_U32_e32
; SI: V_CVT_F32_U32_e32
; SI: S_ENDPGM
define void @uint_to_fp_v2i32(<2 x float> addrspace(1)* %out, <2 x i32> %in) {
  %result = uitofp <2 x i32> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @uint_to_fp_v4i32
; R600: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: V_CVT_F32_U32_e32
; SI: V_CVT_F32_U32_e32
; SI: V_CVT_F32_U32_e32
; SI: V_CVT_F32_U32_e32
; SI: S_ENDPGM
define void @uint_to_fp_v4i32(<4 x float> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %value = load <4 x i32> addrspace(1) * %in
  %result = uitofp <4 x i32> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @uint_to_fp_i64_f32
; R600: UINT_TO_FLT
; R600: UINT_TO_FLT
; R600: MULADD_IEEE
; SI: V_CVT_F32_U32_e32
; SI: V_CVT_F32_U32_e32
; SI: V_MAD_F32
; SI: S_ENDPGM
define void @uint_to_fp_i64_f32(float addrspace(1)* %out, i64 %in) {
entry:
  %0 = uitofp i64 %in to float
  store float %0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @uint_to_fp_i1_f32:
; SI: V_CMP_EQ_I32_e64 [[CMP:s\[[0-9]+:[0-9]\]]],
; SI-NEXT: V_CNDMASK_B32_e64 [[RESULT:v[0-9]+]], 0, 1.000000e+00, [[CMP]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @uint_to_fp_i1_f32(float addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = uitofp i1 %cmp to float
  store float %fp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @uint_to_fp_i1_f32_load:
; SI: V_CNDMASK_B32_e64 [[RESULT:v[0-9]+]], 0, 1.000000e+00
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @uint_to_fp_i1_f32_load(float addrspace(1)* %out, i1 %in) {
  %fp = uitofp i1 %in to float
  store float %fp, float addrspace(1)* %out, align 4
  ret void
}
