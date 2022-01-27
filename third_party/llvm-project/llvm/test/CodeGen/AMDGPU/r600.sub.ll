; RUN: llc -amdgpu-scalarize-global-loads=false -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=EG,FUNC %s

declare i32 @llvm.r600.read.tidig.x() readnone

; FUNC-LABEL: {{^}}s_sub_i32:
define amdgpu_kernel void @s_sub_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %result = sub i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_sub_imm_i32:
define amdgpu_kernel void @s_sub_imm_i32(i32 addrspace(1)* %out, i32 %a) {
  %result = sub i32 1234, %a
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_i32:
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @test_sub_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = sub i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_imm_i32:
; EG: SUB_INT
define amdgpu_kernel void @test_sub_imm_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %a = load i32, i32 addrspace(1)* %in
  %result = sub i32 123, %a
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_v2i32:
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @test_sub_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %b = load <2 x i32>, <2 x i32> addrspace(1) * %b_ptr
  %result = sub <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_v4i32:
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: SUB_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @test_sub_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %b = load <4 x i32>, <4 x i32> addrspace(1) * %b_ptr
  %result = sub <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_i16:
define amdgpu_kernel void @test_sub_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr i16, i16 addrspace(1)* %gep, i32 1
  %a = load volatile i16, i16 addrspace(1)* %gep
  %b = load volatile i16, i16 addrspace(1)* %b_ptr
  %result = sub i16 %a, %b
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_v2i16:
define amdgpu_kernel void @test_sub_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %gep, i16 1
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gep
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b_ptr
  %result = sub <2 x i16> %a, %b
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_sub_v4i16:
define amdgpu_kernel void @test_sub_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %gep, i16 1
  %a = load <4 x i16>, <4 x i16> addrspace(1) * %gep
  %b = load <4 x i16>, <4 x i16> addrspace(1) * %b_ptr
  %result = sub <4 x i16> %a, %b
  store <4 x i16> %result, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_sub_i64:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XY
; EG-DAG: SUB_INT {{[* ]*}}
; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT {{[* ]*}}
define amdgpu_kernel void @s_sub_i64(i64 addrspace(1)* noalias %out, i64 %a, i64 %b) nounwind {
  %result = sub i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_sub_i64:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XY
; EG-DAG: SUB_INT {{[* ]*}}
; EG-DAG: SUBB_UINT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT {{[* ]*}}
define amdgpu_kernel void @v_sub_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %inA, i64 addrspace(1)* noalias %inB) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %a_ptr = getelementptr i64, i64 addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr i64, i64 addrspace(1)* %inB, i32 %tid
  %a = load i64, i64 addrspace(1)* %a_ptr
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = sub i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_test_sub_v2i64:
define amdgpu_kernel void @v_test_sub_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* noalias %inA, <2 x i64> addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %a_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %inB, i32 %tid
  %a = load <2 x i64>, <2 x i64> addrspace(1)* %a_ptr
  %b = load <2 x i64>, <2 x i64> addrspace(1)* %b_ptr
  %result = sub <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_test_sub_v4i64:
define amdgpu_kernel void @v_test_sub_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* noalias %inA, <4 x i64> addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %a_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %inB, i32 %tid
  %a = load <4 x i64>, <4 x i64> addrspace(1)* %a_ptr
  %b = load <4 x i64>, <4 x i64> addrspace(1)* %b_ptr
  %result = sub <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}
