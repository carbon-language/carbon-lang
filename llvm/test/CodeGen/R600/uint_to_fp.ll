; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: @uint_to_fp_v2i32
; R600-CHECK-DAG: UINT_TO_FLT * T{{[0-9]+\.[XYZW]}}, KC0[2].W
; R600-CHECK-DAG: UINT_TO_FLT * T{{[0-9]+\.[XYZW]}}, KC0[3].X
; SI-CHECK: @uint_to_fp_v2i32
; SI-CHECK: V_CVT_F32_U32_e32
; SI-CHECK: V_CVT_F32_U32_e32
define void @uint_to_fp_v2i32(<2 x float> addrspace(1)* %out, <2 x i32> %in) {
  %result = uitofp <2 x i32> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; R600-CHECK: @uint_to_fp_v4i32
; R600-CHECK: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; SI-CHECK: @uint_to_fp_v4i32
; SI-CHECK: V_CVT_F32_U32_e32
; SI-CHECK: V_CVT_F32_U32_e32
; SI-CHECK: V_CVT_F32_U32_e32
; SI-CHECK: V_CVT_F32_U32_e32
define void @uint_to_fp_v4i32(<4 x float> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %value = load <4 x i32> addrspace(1) * %in
  %result = uitofp <4 x i32> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
