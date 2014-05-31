; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: @uint_to_fp_f64_i32
; SI: V_CVT_F64_U32_e32
; SI: S_ENDPGM
define void @uint_to_fp_f64_i32(double addrspace(1)* %out, i32 %in) {
  %cast = uitofp i32 %in to double
  store double %cast, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @uint_to_fp_i1_f64:
; SI: V_CMP_EQ_I32_e64 [[CMP:s\[[0-9]+:[0-9]\]]],
; SI-NEXT: V_CNDMASK_B32_e64 [[IRESULT:v[0-9]+]], 0, 1, [[CMP]]
; SI-NEXT: V_CVT_F64_U32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @uint_to_fp_i1_f64(double addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = uitofp i1 %cmp to double
  store double %fp, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @uint_to_fp_i1_f64_load:
; SI: V_CNDMASK_B32_e64 [[IRESULT:v[0-9]]], 0, 1
; SI-NEXT: V_CVT_F64_U32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
; SI: S_ENDPGM
define void @uint_to_fp_i1_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = uitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}
