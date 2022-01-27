; RUN: llc -march amdgcn %s -filetype=obj -o /dev/null
; RUN: llc -march amdgcn <%s | FileCheck %s
define amdgpu_kernel void @f() {
  ; CHECK: ; divergent unreachable
  call void @llvm.amdgcn.unreachable()
  ret void
}

declare void @llvm.amdgcn.unreachable()
