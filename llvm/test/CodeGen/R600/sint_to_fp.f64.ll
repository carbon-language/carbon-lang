; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; SI-LABEL: {{^}}sint_to_fp_i32_to_f64
; SI: V_CVT_F64_I32_e32
define void @sint_to_fp_i32_to_f64(double addrspace(1)* %out, i32 %in) {
  %result = sitofp i32 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}sint_to_fp_i1_f64:
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

; SI-LABEL: {{^}}sint_to_fp_i1_f64_load:
; SI: V_CNDMASK_B32_e64 [[IRESULT:v[0-9]]], 0, -1
; SI-NEXT: V_CVT_F64_I32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
; SI: S_ENDPGM
define void @sint_to_fp_i1_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = sitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @s_sint_to_fp_i64_to_f64
define void @s_sint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 %in) {
  %result = sitofp i64 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; SI-LABEL: @v_sint_to_fp_i64_to_f64
; SI: BUFFER_LOAD_DWORDX2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; SI-DAG: V_CVT_F64_U32_e32 [[LO_CONV:v\[[0-9]+:[0-9]+\]]], v[[LO]]
; SI-DAG: V_CVT_F64_I32_e32 [[HI_CONV:v\[[0-9]+:[0-9]+\]]], v[[HI]]
; SI: V_LDEXP_F64 [[LDEXP:v\[[0-9]+:[0-9]+\]]], [[HI_CONV]], 32
; SI: V_ADD_F64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[LDEXP]], [[LO_CONV]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
define void @v_sint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep = getelementptr i64 addrspace(1)* %in, i32 %tid
  %val = load i64 addrspace(1)* %gep, align 8
  %result = sitofp i64 %val to double
  store double %result, double addrspace(1)* %out
  ret void
}
