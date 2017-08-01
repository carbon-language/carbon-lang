; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -amdgpu-sroa=0 -amdgpu-function-calls -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=hawaii -amdgpu-function-calls -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-sroa=0 -amdgpu-function-calls -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GFX9 -check-prefix=VI %s

; Test calls when called by other callable functions rather than
; kernels.

declare void @external_void_func_i32(i32) #0

; GCN-LABEL: {{^}}test_func_call_external_void_func_i32_imm:
; GCN: s_waitcnt
; GCN-NOT: s32
; GCN: s_swappc_b64
; GCN-NOT: s32
; GCN: s_setpc_b64
define void @test_func_call_external_void_func_i32_imm() #0 {
  call void @external_void_func_i32(i32 42)
  ret void
}

; GCN-LABEL: {{^}}test_func_call_external_void_func_i32_imm_stack_use:
; GCN: s_waitcnt
; GCN: s_mov_b32 s5, s32
; GCN: s_add_u32 s32, s32, 0x1100{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s5 offset
; GCN: s_swappc_b64
; GCN: s_sub_u32 s32, s32, 0x1100{{$}}
; GCN: s_setpc_b64
define void @test_func_call_external_void_func_i32_imm_stack_use() #0 {
  %alloca = alloca [16 x i32], align 4
  %gep0 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 0
  %gep15 = getelementptr inbounds [16 x i32], [16 x i32]* %alloca, i32 0, i32 16
  store volatile i32 0, i32* %gep0
  store volatile i32 0, i32* %gep15
  call void @external_void_func_i32(i32 42)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
