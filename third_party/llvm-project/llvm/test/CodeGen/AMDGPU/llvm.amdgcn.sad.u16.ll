; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.sad.u16(i32, i32, i32) #0

; GCN-LABEL: {{^}}v_sad_u16:
; GCN: v_sad_u16 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @v_sad_u16(i32 addrspace(1)* %out, i32 %src) {
  %result= call i32 @llvm.amdgcn.sad.u16(i32 %src, i32 100, i32 100) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_sad_u16_non_immediate:
; GCN: v_sad_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u16_non_immediate(i32 addrspace(1)* %out, i32 %src, i32 %a, i32 %b) {
  %result= call i32 @llvm.amdgcn.sad.u16(i32 %src, i32 %a, i32 %b) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
