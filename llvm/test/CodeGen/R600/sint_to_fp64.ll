; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI: @sint_to_fp64
; SI: V_CVT_F64_I32_e32
define void @sint_to_fp64(double addrspace(1)* %out, i32 %in) {
  %result = sitofp i32 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; SI-LABEL: @sint_to_fp_i1_f64:
; SI: V_CMP_EQ_I32_e64 [[CMP:s\[[0-9]+:[0-9]\]]],
; FIXME: We should the VGPR sources for V_CNDMASK are copied from SGPRs,
; we should be able to fold the SGPRs into the V_CNDMASK instructions.
; SI: V_CNDMASK_B32_e64 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CMP]]
; SI: V_CNDMASK_B32_e64 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CMP]]
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
define void @sint_to_fp_i1_f64(double addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = sitofp i1 %cmp to double
  store double %fp, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @sint_to_fp_i1_f64_load:
; SI: V_CNDMASK_B32_e64 [[IRESULT:v[0-9]]], 0, -1
; SI-NEXT: V_CVT_F64_I32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
; SI: S_ENDPGM
define void @sint_to_fp_i1_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = sitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}
