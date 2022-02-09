; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.cubesc(float, float, float) #0

; GCN-LABEL: {{^}}test_cubesc:
; GCN: v_cubesc_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_cubesc(float addrspace(1)* %out, float %a, float %b, float %c) #1 {
  %result = call float @llvm.amdgcn.cubesc(float %a, float %b, float %c)
  store float %result, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
