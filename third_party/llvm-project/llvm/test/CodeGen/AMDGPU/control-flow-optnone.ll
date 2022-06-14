; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; optnone disables AMDGPUAnnotateUniformValues, so no branch is known
; to be uniform during instruction selection. The custom selection for
; brcond was not checking if the branch was uniform, relying on the
; selection pattern to check that. That would fail, so then the branch
; would fail to select.

; GCN-LABEL: {{^}}copytoreg_divergent_brcond:
; GCN: s_branch

; GCN-DAG: v_cmp_lt_i32
; GCN-DAG: s_cmp_gt_i32
; GCN: s_and_b64
; GCN: s_mov_b64 exec

; GCN: s_or_b64 exec, exec
; GCN: {{[s|v]}}_cmp_eq_u32
; GCN: s_cbranch
; GCN-NEXT: s_branch
define amdgpu_kernel void @copytoreg_divergent_brcond(i32 %arg, i32 %arg1, i32 %arg2) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = zext i32 %tmp to i64
  %tmp5 = add i64 %tmp3, undef
  %tmp6 = trunc i64 %tmp5 to i32
  %tmp7 = mul nsw i32 %tmp6, %arg2
  br label %bb8

bb8.loopexit:                                     ; preds = %bb14
  br label %bb8

bb8:                                              ; preds = %bb8.loopexit, %bb
  br label %bb9

bb9:                                              ; preds = %bb14, %bb8
  %tmp10 = icmp slt i32 %tmp7, %arg1
  %tmp11 = icmp sgt i32 %arg, 0
  %tmp12 = and i1 %tmp10, %tmp11
  br i1 %tmp12, label %bb13, label %bb14

bb13:                                             ; preds = %bb9
  store volatile i32 0, i32 addrspace(1)* undef, align 4
  br label %bb14

bb14:                                             ; preds = %bb13, %bb9
  %tmp15 = icmp eq i32 %arg2, 1
  br i1 %tmp15, label %bb8.loopexit, label %bb9
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind optnone noinline }
attributes #1 = { nounwind readnone speculatable }
