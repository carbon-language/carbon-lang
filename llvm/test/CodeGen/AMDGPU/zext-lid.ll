; RUN: llc -march=amdgcn < %s | FileCheck -enable-var-scope -check-prefixes=GCN,O2 %s
; RUN: llc -O0 -march=amdgcn < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-intrinsics < %s | FileCheck -check-prefix=OPT %s

; GCN-LABEL: {{^}}zext_grp_size_128:
; GCN-NOT: and_b32

; OPT-LABEL: @zext_grp_size_128
; OPT: tail call i32 @llvm.amdgcn.workitem.id.x(), !range !0
; OPT: tail call i32 @llvm.amdgcn.workitem.id.y(), !range !0
; OPT: tail call i32 @llvm.amdgcn.workitem.id.z(), !range !0
define amdgpu_kernel void @zext_grp_size_128(i32 addrspace(1)* nocapture %arg) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = and i32 %tmp, 127
  store i32 %tmp1, i32 addrspace(1)* %arg, align 4
  %tmp2 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp3 = and i32 %tmp2, 127
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  store i32 %tmp3, i32 addrspace(1)* %tmp4, align 4
  %tmp5 = tail call i32 @llvm.amdgcn.workitem.id.z()
  %tmp6 = and i32 %tmp5, 127
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 2
  store i32 %tmp6, i32 addrspace(1)* %tmp7, align 4
  ret void
}

; GCN-LABEL: {{^}}zext_grp_size_32x4x1:
; GCN-NOT: and_b32

; OPT-LABEL: @zext_grp_size_32x4x1
; OPT: tail call i32 @llvm.amdgcn.workitem.id.x(), !range !2
; OPT: tail call i32 @llvm.amdgcn.workitem.id.y(), !range !3
; OPT: tail call i32 @llvm.amdgcn.workitem.id.z(), !range !4
define amdgpu_kernel void @zext_grp_size_32x4x1(i32 addrspace(1)* nocapture %arg) #0 !reqd_work_group_size !0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = and i32 %tmp, 31
  store i32 %tmp1, i32 addrspace(1)* %arg, align 4
  %tmp2 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp3 = and i32 %tmp2, 3
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  store i32 %tmp3, i32 addrspace(1)* %tmp4, align 4
  %tmp5 = tail call i32 @llvm.amdgcn.workitem.id.z()
  %tmp6 = and i32 %tmp5, 1
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 2
  store i32 %tmp6, i32 addrspace(1)* %tmp7, align 4
  ret void
}

; GCN-LABEL: {{^}}zext_grp_size_1x1x1:
; GCN-NOT: and_b32

; When EarlyCSE is not run this call produces a range max with 0 active bits,
; which is a special case as an AssertZext from width 0 is invalid.
; OPT-LABEL: @zext_grp_size_1x1x1
; OPT: tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
define amdgpu_kernel void @zext_grp_size_1x1x1(i32 addrspace(1)* nocapture %arg) #0 !reqd_work_group_size !1 {
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = and i32 %tmp, 1
  store i32 %tmp1, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}zext_grp_size_512:
; GCN-NOT: and_b32

; OPT-LABEL: @zext_grp_size_512
; OPT: tail call i32 @llvm.amdgcn.workitem.id.x(), !range !6
; OPT: tail call i32 @llvm.amdgcn.workitem.id.y(), !range !6
; OPT: tail call i32 @llvm.amdgcn.workitem.id.z(), !range !6
define amdgpu_kernel void @zext_grp_size_512(i32 addrspace(1)* nocapture %arg) #1 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = and i32 %tmp, 65535
  store i32 %tmp1, i32 addrspace(1)* %arg, align 4
  %tmp2 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp3 = and i32 %tmp2, 65535
  %tmp4 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  store i32 %tmp3, i32 addrspace(1)* %tmp4, align 4
  %tmp5 = tail call i32 @llvm.amdgcn.workitem.id.z()
  %tmp6 = and i32 %tmp5, 65535
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 2
  store i32 %tmp6, i32 addrspace(1)* %tmp7, align 4
  ret void
}

; GCN-LABEL: {{^}}func_test_workitem_id_x_known_max_range:
; O2-NOT: and_b32
; O2: v_and_b32_e32 v{{[0-9]+}}, 0x3ff,
; O2-NOT: and_b32

; OPT-LABEL: @func_test_workitem_id_x_known_max_range(
; OPT: tail call i32 @llvm.amdgcn.workitem.id.x(), !range !0
define void @func_test_workitem_id_x_known_max_range(i32 addrspace(1)* nocapture %out) #0 {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %id, 1023
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}func_test_workitem_id_x_default_range:
; O2-NOT: and_b32
; O2: v_and_b32_e32 v{{[0-9]+}}, 0x3ff,
; O2-NOT: and_b32

; OPT-LABEL: @func_test_workitem_id_x_default_range(
; OPT: tail call i32 @llvm.amdgcn.workitem.id.x(), !range !7
define void @func_test_workitem_id_x_default_range(i32 addrspace(1)* nocapture %out) #4 {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %id, 1023
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #2

declare i32 @llvm.amdgcn.workitem.id.y() #2

declare i32 @llvm.amdgcn.workitem.id.z() #2

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,128" }
attributes #1 = { nounwind "amdgpu-flat-work-group-size"="512,512" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!0 = !{i32 32, i32 4, i32 1}
!1 = !{i32 1, i32 1, i32 1}

; OPT: !0 = !{i32 0, i32 128}
; OPT: !1 = !{i32 32, i32 4, i32 1}
; OPT: !2 = !{i32 0, i32 32}
; OPT: !3 = !{i32 0, i32 4}
; OPT: !4 = !{i32 0, i32 1}
; OPT: !5 = !{i32 1, i32 1, i32 1}
; OPT: !6 = !{i32 0, i32 512}
; OPT: !7 = !{i32 0, i32 1024}
