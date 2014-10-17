; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() #1
declare float @llvm.fabs.f32(float) #1

; FUNC-LABEL: @commute_add_imm_fabs_f32
; SI: BUFFER_LOAD_DWORD [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: V_ADD_F32_e64 [[REG:v[0-9]+]], 2.0, |[[X]]|
; SI-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @commute_add_imm_fabs_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float addrspace(1)* %in, i32 %tid
  %x = load float addrspace(1)* %gep.0
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %z = fadd float 2.0, %x.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_imm_fneg_fabs_f32
; SI: BUFFER_LOAD_DWORD [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: V_MUL_F32_e64 [[REG:v[0-9]+]], -4.0, |[[X]]|
; SI-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @commute_mul_imm_fneg_fabs_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float addrspace(1)* %in, i32 %tid
  %x = load float addrspace(1)* %gep.0
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %x.fneg.fabs = fsub float -0.000000e+00, %x.fabs
  %z = fmul float 4.0, %x.fneg.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_imm_fneg_f32
; SI: BUFFER_LOAD_DWORD [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: V_MUL_F32_e32 [[REG:v[0-9]+]], -4.0, [[X]]
; SI-NEXT: BUFFER_STORE_DWORD [[REG]]
define void @commute_mul_imm_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float addrspace(1)* %in, i32 %tid
  %x = load float addrspace(1)* %gep.0
  %x.fneg = fsub float -0.000000e+00, %x
  %z = fmul float 4.0, %x.fneg
  store float %z, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
