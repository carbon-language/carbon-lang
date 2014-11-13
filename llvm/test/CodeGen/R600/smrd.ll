; RUN: llc < %s -march=r600 -mcpu=SI -show-mc-encoding -verify-machineinstrs | FileCheck %s

; SMRD load with an immediate offset.
; CHECK-LABEL: {{^}}smrd0:
; CHECK: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x1 ; encoding: [0x01
define void @smrd0(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) {
entry:
  %0 = getelementptr i32 addrspace(2)* %ptr, i64 1
  %1 = load i32 addrspace(2)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with the largest possible immediate offset.
; CHECK-LABEL: {{^}}smrd1:
; CHECK: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff
define void @smrd1(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) {
entry:
  %0 = getelementptr i32 addrspace(2)* %ptr, i64 255
  %1 = load i32 addrspace(2)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with an offset greater than the largest possible immediate.
; CHECK-LABEL: {{^}}smrd2:
; CHECK: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; CHECK: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CHECK: s_endpgm
define void @smrd2(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) {
entry:
  %0 = getelementptr i32 addrspace(2)* %ptr, i64 256
  %1 = load i32 addrspace(2)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with a 64-bit offset
; CHECK-LABEL: {{^}}smrd3:
; CHECK-DAG: s_mov_b32 s[[SHI:[0-9]+]], 4
; CHECK-DAG: s_mov_b32 s[[SLO:[0-9]+]], 0 ;
; FIXME: We don't need to copy these values to VGPRs
; CHECK-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; CHECK-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; FIXME: We should be able to use s_load_dword here
; CHECK: buffer_load_dword v{{[0-9]+}}, v{{\[}}[[VLO]]:[[VHI]]{{\]}}, s[{{[0-9]+:[0-9]+}}], 0 addr64
; CHECK: s_endpgm
define void @smrd3(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) {
entry:
  %0 = getelementptr i32 addrspace(2)* %ptr, i64 4294967296 ; 2 ^ 32
  %1 = load i32 addrspace(2)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; SMRD load using the load.const intrinsic with an immediate offset
; CHECK-LABEL: {{^}}smrd_load_const0:
; CHECK: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4 ; encoding: [0x04
define void @smrd_load_const0(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr <16 x i8> addrspace(2)* %0, i32 0
  %21 = load <16 x i8> addrspace(2)* %20
  %22 = call float @llvm.SI.load.const(<16 x i8> %21, i32 16)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %22, float %22, float %22, float %22)
  ret void
}

; SMRD load using the load.const intrinsic with the largest possible immediate
; offset.
; CHECK-LABEL: {{^}}smrd_load_const1:
; CHECK: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff
define void @smrd_load_const1(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr <16 x i8> addrspace(2)* %0, i32 0
  %21 = load <16 x i8> addrspace(2)* %20
  %22 = call float @llvm.SI.load.const(<16 x i8> %21, i32 1020)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %22, float %22, float %22, float %22)
  ret void
}
; SMRD load using the load.const intrinsic with an offset greater than the
; largets possible immediate.
; immediate offset.
; CHECK-LABEL: {{^}}smrd_load_const2:
; CHECK: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; CHECK: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
define void @smrd_load_const2(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr <16 x i8> addrspace(2)* %0, i32 0
  %21 = load <16 x i8> addrspace(2)* %20
  %22 = call float @llvm.SI.load.const(<16 x i8> %21, i32 1024)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %22, float %22, float %22, float %22)
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
