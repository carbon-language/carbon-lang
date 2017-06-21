; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}add1:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_addc_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, 0, v{{[0-9]+}}, [[CC]]
; GCN-NOT: v_cndmask

define amdgpu_kernel void @add1(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = zext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}sub1:
; GCN: v_cmp_gt_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_subb_u32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, v{{[0-9]+}}, 0, [[CC]]
; GCN-NOT: v_cndmask

define amdgpu_kernel void @sub1(i32 addrspace(1)* nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %x
  %v = load i32, i32 addrspace(1)* %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %add = add i32 %v, %ext
  store i32 %add, i32 addrspace(1)* %gep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workitem.id.y() #0

attributes #0 = { nounwind readnone speculatable }
