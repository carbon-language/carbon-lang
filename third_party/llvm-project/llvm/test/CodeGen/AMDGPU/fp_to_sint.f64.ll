; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare double @llvm.fabs.f64(double) #1

; FUNC-LABEL: @fp_to_sint_f64_i32
; SI: v_cvt_i32_f64_e32
define amdgpu_kernel void @fp_to_sint_f64_i32(i32 addrspace(1)* %out, double %in) {
  %result = fptosi double %in to i32
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fp_to_sint_v2f64_v2i32
; SI: v_cvt_i32_f64_e32
; SI: v_cvt_i32_f64_e32
define amdgpu_kernel void @fp_to_sint_v2f64_v2i32(<2 x i32> addrspace(1)* %out, <2 x double> %in) {
  %result = fptosi <2 x double> %in to <2 x i32>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fp_to_sint_v4f64_v4i32
; SI: v_cvt_i32_f64_e32
; SI: v_cvt_i32_f64_e32
; SI: v_cvt_i32_f64_e32
; SI: v_cvt_i32_f64_e32
define amdgpu_kernel void @fp_to_sint_v4f64_v4i32(<4 x i32> addrspace(1)* %out, <4 x double> %in) {
  %result = fptosi <4 x double> %in to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fp_to_sint_i64_f64
; CI-DAG: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; CI-DAG: v_trunc_f64_e32 [[TRUNC:v\[[0-9]+:[0-9]+\]]], [[VAL]]
; CI-DAG: s_mov_b32 s[[K0_LO:[0-9]+]], 0{{$}}
; CI-DAG: s_mov_b32 s[[K0_HI:[0-9]+]], 0x3df00000

; CI-DAG: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], [[VAL]], s{{\[}}[[K0_LO]]:[[K0_HI]]{{\]}}
; CI-DAG: v_floor_f64_e32 [[FLOOR:v\[[0-9]+:[0-9]+\]]], [[MUL]]

; CI-DAG: s_mov_b32 s[[K1_HI:[0-9]+]], 0xc1f00000

; CI-DAG: v_fma_f64 [[FMA:v\[[0-9]+:[0-9]+\]]], [[FLOOR]], s{{\[[0-9]+}}:[[K1_HI]]{{\]}}, [[TRUNC]]
; CI-DAG: v_cvt_u32_f64_e32 v[[LO:[0-9]+]], [[FMA]]
; CI-DAG: v_cvt_i32_f64_e32 v[[HI:[0-9]+]], [[FLOOR]]
; CI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @fp_to_sint_i64_f64(i64 addrspace(1)* %out, double addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid
  %val = load double, double addrspace(1)* %gep, align 8
  %cast = fptosi double %val to i64
  store i64 %cast, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_f64_to_i1:
; SI: v_cmp_eq_f64_e64 s{{\[[0-9]+:[0-9]+\]}}, -1.0, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @fp_to_sint_f64_to_i1(i1 addrspace(1)* %out, double %in) #0 {
  %conv = fptosi double %in to i1
  store i1 %conv, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_fabs_f64_to_i1:
; SI: v_cmp_eq_f64_e64 s{{\[[0-9]+:[0-9]+\]}}, -1.0, |s{{\[[0-9]+:[0-9]+\]}}|
define amdgpu_kernel void @fp_to_sint_fabs_f64_to_i1(i1 addrspace(1)* %out, double %in) #0 {
  %in.fabs = call double @llvm.fabs.f64(double %in)
  %conv = fptosi double %in.fabs to i1
  store i1 %conv, i1 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
