; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: @sint_to_fp_v4i32
; R600-CHECK: INT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: INT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: INT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-CHECK: INT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; SI-CHECK: @sint_to_fp_v4i32
; SI-CHECK: V_CVT_F32_I32_e32
; SI-CHECK: V_CVT_F32_I32_e32
; SI-CHECK: V_CVT_F32_I32_e32
; SI-CHECK: V_CVT_F32_I32_e32
define void @sint_to_fp_v4i32(<4 x float> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %value = load <4 x i32> addrspace(1) * %in
  %result = sitofp <4 x i32> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
