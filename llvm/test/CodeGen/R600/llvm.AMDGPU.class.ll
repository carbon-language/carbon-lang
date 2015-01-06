; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i1 @llvm.AMDGPU.class.f32(float, i32) #1
declare i1 @llvm.AMDGPU.class.f64(double, i32) #1
declare i32 @llvm.r600.read.tidig.x() #1
declare float @llvm.fabs.f32(float) #1
declare double @llvm.fabs.f64(double) #1

; SI-LABEL: {{^}}test_class_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e32 vcc, [[SA]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %result = call i1 @llvm.AMDGPU.class.f32(float %a, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fabs_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], |[[SA]]|, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_fabs_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %a.fabs = call float @llvm.fabs.f32(float %a) #1
  %result = call i1 @llvm.AMDGPU.class.f32(float %a.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -[[SA]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_fneg_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %a.fneg = fsub float -0.0, %a
  %result = call i1 @llvm.AMDGPU.class.f32(float %a.fneg, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_fabs_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -|[[SA]]|, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_fneg_fabs_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %a.fabs = call float @llvm.fabs.f32(float %a) #1
  %a.fneg.fabs = fsub float -0.0, %a.fabs
  %result = call i1 @llvm.AMDGPU.class.f32(float %a.fneg.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_1_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_cmp_class_f32_e64 [[COND:s\[[0-9]+:[0-9]+\]]], [[SA]], 1{{$}}
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[COND]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_1_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f32(float %a, i32 1) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_64_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_cmp_class_f32_e64 [[COND:s\[[0-9]+:[0-9]+\]]], [[SA]], 64{{$}}
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[COND]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_64_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f32(float %a, i32 64) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; Set all 10 bits of mask
; SI-LABEL: {{^}}test_class_full_mask_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x3ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, [[SA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_full_mask_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f32(float %a, i32 1023) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_9bit_mask_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, [[SA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_9bit_mask_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f32(float %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}v_test_class_full_mask_f32:
; SI-DAG: buffer_load_dword [[VA:v[0-9]+]]
; SI-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, [[VA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @v_test_class_full_mask_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.in = getelementptr float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32 addrspace(1)* %out, i32 %tid
  %a = load float addrspace(1)* %gep.in

  %result = call i1 @llvm.AMDGPU.class.f32(float %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_inline_imm_constant_dynamic_mask_f32:
; SI-DAG: buffer_load_dword [[VB:v[0-9]+]]
; SI: v_cmp_class_f32_e32 vcc, 1.0, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_inline_imm_constant_dynamic_mask_f32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.in = getelementptr i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32 addrspace(1)* %out, i32 %tid
  %b = load i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.AMDGPU.class.f32(float 1.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; FIXME: Why isn't this using a literal constant operand?
; SI-LABEL: {{^}}test_class_lit_constant_dynamic_mask_f32:
; SI-DAG: buffer_load_dword [[VB:v[0-9]+]]
; SI-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; SI: v_cmp_class_f32_e32 vcc, [[VK]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_lit_constant_dynamic_mask_f32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.in = getelementptr i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32 addrspace(1)* %out, i32 %tid
  %b = load i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.AMDGPU.class.f32(float 1024.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e32 vcc, [[SA]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %result = call i1 @llvm.AMDGPU.class.f64(double %a, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fabs_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], |[[SA]]|, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_fabs_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %a.fabs = call double @llvm.fabs.f64(double %a) #1
  %result = call i1 @llvm.AMDGPU.class.f64(double %a.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -[[SA]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_fneg_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %a.fneg = fsub double -0.0, %a
  %result = call i1 @llvm.AMDGPU.class.f64(double %a.fneg, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_fabs_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -|[[SA]]|, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_fneg_fabs_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %a.fabs = call double @llvm.fabs.f64(double %a) #1
  %a.fneg.fabs = fsub double -0.0, %a.fabs
  %result = call i1 @llvm.AMDGPU.class.f64(double %a.fneg.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_1_f64:
; SI: v_cmp_class_f64_e64 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 1{{$}}
; SI: s_endpgm
define void @test_class_1_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f64(double %a, i32 1) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_64_f64:
; SI: v_cmp_class_f64_e64 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 64{{$}}
; SI: s_endpgm
define void @test_class_64_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f64(double %a, i32 64) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; Set all 9 bits of mask
; SI-LABEL: {{^}}test_class_full_mask_f64:
; SI: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f64_e32 vcc, [[SA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @test_class_full_mask_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.AMDGPU.class.f64(double %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}v_test_class_full_mask_f64:
; SI-DAG: buffer_load_dwordx2 [[VA:v\[[0-9]+:[0-9]+\]]]
; SI-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f64_e32 vcc, [[VA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define void @v_test_class_full_mask_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.in = getelementptr double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32 addrspace(1)* %out, i32 %tid
  %a = load double addrspace(1)* %in

  %result = call i1 @llvm.AMDGPU.class.f64(double %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_inline_imm_constant_dynamic_mask_f64:
; XSI: v_cmp_class_f64_e32 vcc, 1.0,
; SI: v_cmp_class_f64_e32 vcc,
; SI: s_endpgm
define void @test_class_inline_imm_constant_dynamic_mask_f64(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.in = getelementptr i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32 addrspace(1)* %out, i32 %tid
  %b = load i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.AMDGPU.class.f64(double 1.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_lit_constant_dynamic_mask_f64:
; SI: v_cmp_class_f64_e32 vcc, s{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}
; SI: s_endpgm
define void @test_class_lit_constant_dynamic_mask_f64(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.in = getelementptr i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32 addrspace(1)* %out, i32 %tid
  %b = load i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.AMDGPU.class.f64(double 1024.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
