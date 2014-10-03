; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; SI-LABEL: {{^}}uint_to_fp_f64_i32
; SI: V_CVT_F64_U32_e32
; SI: S_ENDPGM
define void @uint_to_fp_f64_i32(double addrspace(1)* %out, i32 %in) {
  %cast = uitofp i32 %in to double
  store double %cast, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}uint_to_fp_i1_f64:
; SI: V_CMP_EQ_I32_e64 [[CMP:s\[[0-9]+:[0-9]\]]],
; FIXME: We should the VGPR sources for V_CNDMASK are copied from SGPRs,
; we should be able to fold the SGPRs into the V_CNDMASK instructions.
; SI: V_CNDMASK_B32_e64 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CMP]]
; SI: V_CNDMASK_B32_e64 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CMP]]
; SI: BUFFER_STORE_DWORDX2
; SI: S_ENDPGM
define void @uint_to_fp_i1_f64(double addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = uitofp i1 %cmp to double
  store double %fp, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}uint_to_fp_i1_f64_load:
; SI: V_CNDMASK_B32_e64 [[IRESULT:v[0-9]]], 0, 1
; SI-NEXT: V_CVT_F64_U32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
; SI: S_ENDPGM
define void @uint_to_fp_i1_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = uitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}v_uint_to_fp_i64_to_f64
; SI: BUFFER_LOAD_DWORDX2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; SI-DAG: V_CVT_F64_U32_e32 [[LO_CONV:v\[[0-9]+:[0-9]+\]]], v[[LO]]
; SI-DAG: V_CVT_F64_U32_e32 [[HI_CONV:v\[[0-9]+:[0-9]+\]]], v[[HI]]
; SI: V_LDEXP_F64 [[LDEXP:v\[[0-9]+:[0-9]+\]]], [[HI_CONV]], 32
; SI: V_ADD_F64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[LDEXP]], [[LO_CONV]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
define void @v_uint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep = getelementptr i64 addrspace(1)* %in, i32 %tid
  %val = load i64 addrspace(1)* %gep, align 8
  %result = uitofp i64 %val to double
  store double %result, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_f64_i64
define void @s_uint_to_fp_f64_i64(double addrspace(1)* %out, i64 %in) {
  %cast = uitofp i64 %in to double
  store double %cast, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_v2f64_v2i64
define void @s_uint_to_fp_v2f64_v2i64(<2 x double> addrspace(1)* %out, <2 x i64> %in) {
  %cast = uitofp <2 x i64> %in to <2 x double>
  store <2 x double> %cast, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_v4f64_v4i64
define void @s_uint_to_fp_v4f64_v4i64(<4 x double> addrspace(1)* %out, <4 x i64> %in) {
  %cast = uitofp <4 x i64> %in to <4 x double>
  store <4 x double> %cast, <4 x double> addrspace(1)* %out, align 16
  ret void
}
