; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

declare float @llvm.fma.f32(float, float, float) #1
declare double @llvm.fma.f64(double, double, double) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare i32 @llvm.AMDGPU.imad24(i32, i32, i32) #1


; GCN-LABEL: {{^}}test_sgpr_use_twice_binop:
; GCN: s_load_dword [[SGPR:s[0-9]+]],
; GCN: v_add_f32_e64 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_binop(float addrspace(1)* %out, float %a) #0 {
  %dbl = fadd float %a, %a
  store float %dbl, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_three_ternary_op:
; GCN: s_load_dword [[SGPR:s[0-9]+]],
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_three_ternary_op(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_a_b:
; SI: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; VI: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x30
; GCN: v_mov_b32_e32 [[VGPR1:v[0-9]+]], [[SGPR1]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR0]], [[SGPR0]], [[VGPR1]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_a_a_b(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float %b) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_b_a:
; SI: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; VI: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x30
; GCN: v_mov_b32_e32 [[VGPR1:v[0-9]+]], [[SGPR1]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[VGPR1]], [[SGPR0]], [[SGPR0]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_a_b_a(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %b, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_b_a_a:
; SI: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; VI: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x30
; GCN: v_mov_b32_e32 [[VGPR1:v[0-9]+]], [[SGPR1]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR0]], [[VGPR1]], [[SGPR0]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_b_a_a(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma = call float @llvm.fma.f32(float %b, float %a, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_a_imm:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]], 2.0
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_a_a_imm(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float 2.0) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_imm_a:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], 2.0, [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_a_imm_a(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float 2.0, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; Don't use fma since fma c, x, y is canonicalized to fma x, c, y
; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_imm_a_a:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_mad_i32_i24 [[RESULT:v[0-9]+]], 2, [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_imm_a_a(i32 addrspace(1)* %out, i32 %a) #0 {
  %fma = call i32 @llvm.AMDGPU.imad24(i32 2, i32 %a, i32 %a) #1
  store i32 %fma, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_a_kimm:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]], [[VK]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_sgpr_use_twice_ternary_op_a_a_kimm(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float 1024.0) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_k_s:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT0:v[0-9]+]], [[VK]], [[VK]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT0]]
define void @test_literal_use_twice_ternary_op_k_k_s(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float 1024.0, float 1024.0, float %a) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_k_s_x2:
; GCN-DAG: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[VK]], [[VK]], [[SGPR0]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[VK]], [[VK]], [[SGPR1]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
; GCN: s_endpgm
define void @test_literal_use_twice_ternary_op_k_k_s_x2(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float 1024.0, float 1024.0, float %a) #1
  %fma1 = call float @llvm.fma.f32(float 1024.0, float 1024.0, float %b) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_s_k:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[VK]], [[VK]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_literal_use_twice_ternary_op_k_s_k(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float 1024.0, float %a, float 1024.0) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_s_k_x2:
; GCN-DAG: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[SGPR0]], [[VK]], [[VK]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[SGPR1]], [[VK]], [[VK]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
; GCN: s_endpgm
define void @test_literal_use_twice_ternary_op_k_s_k_x2(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float 1024.0, float %a, float 1024.0) #1
  %fma1 = call float @llvm.fma.f32(float 1024.0, float %b, float 1024.0) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_s_k_k:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[VK]], [[VK]]
; GCN: buffer_store_dword [[RESULT]]
define void @test_literal_use_twice_ternary_op_s_k_k(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float 1024.0, float 1024.0) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_s_k_k_x2:
; GCN-DAG: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[SGPR0]], [[VK]], [[VK]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[SGPR1]], [[VK]], [[VK]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
; GCN: s_endpgm
define void @test_literal_use_twice_ternary_op_s_k_k_x2(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float %a, float 1024.0, float 1024.0) #1
  %fma1 = call float @llvm.fma.f32(float %b, float 1024.0, float 1024.0) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_s0_s1_k_f32:
; GCN-DAG: s_load_dword [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[SGPR1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN-DAG: v_mov_b32_e32 [[VK0:v[0-9]+]], 0x44800000

; FIXME: Why do we end up with 2 copies of the same SGPR? These should be CSE'd
; GCN: v_mov_b32_e32 [[VS1_1:v[0-9]+]], [[SGPR1]]
; GCN: v_mov_b32_e32 [[VS1_0:v[0-9]+]], [[SGPR1]]

; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[SGPR0]], [[VS1_0]], [[VK0]]
; GCN-DAG: v_mov_b32_e32 [[VK1:v[0-9]+]], 0x45800000
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[SGPR0]], [[VS1_1]], [[VK1]]

; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
define void @test_s0_s1_k_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float %a, float %b, float 1024.0) #1
  %fma1 = call float @llvm.fma.f32(float %a, float %b, float 4096.0) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; FIXME: Immediate in SGPRs just copied to VGPRs
; GCN-LABEL: {{^}}test_s0_s1_k_f64:
; GCN-DAG: s_load_dwordx2 [[SGPR0:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dwordx2 s{{\[}}[[SGPR1_SUB0:[0-9]+]]:[[SGPR1_SUB1:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}
; GCN-DAG: s_mov_b32 s[[SK0_SUB1:[0-9]+]], 0x40900000
; GCN-DAG: s_mov_b32 s[[SZERO:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[VK0_SUB0:[0-9]+]], s[[SZERO]]
; GCN-DAG: v_mov_b32_e32 v[[VK0_SUB1:[0-9]+]], s[[SK0_SUB1]]

; GCN-DAG: s_mov_b32 s[[SK1_SUB0:[0-9]+]], 0x40b00000{{$}}

; FIXME: Redundant copies
; GCN: v_mov_b32_e32 v[[VS1_1_SUB0:[0-9]+]], s[[SGPR1_SUB0]]
; GCN: v_mov_b32_e32 v[[VS1_1_SUB1:[0-9]+]], s[[SGPR1_SUB1]]
; GCN: v_mov_b32_e32 v[[VS1_0_SUB0:[0-9]+]], s[[SGPR1_SUB0]]
; GCN: v_mov_b32_e32 v[[VS1_0_SUB1:[0-9]+]], s[[SGPR1_SUB1]]


; GCN-DAG: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[SGPR0]], v{{\[}}[[VS1_0_SUB0]]:[[VS1_0_SUB1]]{{\]}}, v{{\[}}[[VK0_SUB0]]:[[VK0_SUB1]]{{\]}}

; GCN-DAG: v_mov_b32_e32 v[[VK1_SUB0:[0-9]+]], s[[SZERO]]
; GCN-DAG: v_mov_b32_e32 v[[VK1_SUB1:[0-9]+]], s[[SK1_SUB0]]

; GCN-DAG: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[SGPR0]], v{{\[}}[[VS1_1_SUB0]]:[[VS1_1_SUB1]]{{\]}}, v{{\[}}[[VK1_SUB0]]:[[VK1_SUB1]]{{\]}}

; GCN: buffer_store_dwordx2 [[RESULT0]]
; GCN: buffer_store_dwordx2 [[RESULT1]]
define void @test_s0_s1_k_f64(double addrspace(1)* %out, double %a, double %b) #0 {
  %fma0 = call double @llvm.fma.f64(double %a, double %b, double 1024.0) #1
  %fma1 = call double @llvm.fma.f64(double %a, double %b, double 4096.0) #1
  store volatile double %fma0, double addrspace(1)* %out
  store volatile double %fma1, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
