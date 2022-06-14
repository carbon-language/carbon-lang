; RUN:  llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN:  llc -march=amdgcn -mcpu=gfx900 -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX900 %s

; FUNC-LABEL: {{^}}uniform_add_SIC:
; GCN: S_SUB_I32 killed %{{[0-9]+}}, 32
define amdgpu_kernel void @uniform_add_SIC(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %a = load i32, i32 addrspace(1)* %in
  %result = add i32 %a, -32
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}uniform_add_SIC:
; SI: V_SUB_CO_U32_e64 killed %{{[0-9]+}}, 32
; GFX900: V_SUB_U32_e64 killed %{{[0-9]+}}, 32
define amdgpu_kernel void @divergent_add_SIC(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %tid
  %a = load volatile i32, i32 addrspace(1)* %gep
  %result = add i32 %a, -32
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
