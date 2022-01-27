; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx1010 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK

; This used to cause a circular chain dependency during
; SelectionDAG instruction scheduling.

; CHECK-LABEL: {{^}}_amdgpu_gs_main:
; CHECK: ds_read_b32
; CHECK: ds_read_b32
; CHECK: ds_read_b32
; CHECK: ds_read_b32
define amdgpu_gs float @_amdgpu_gs_main(i8 addrspace(3)* %arg0, i8 addrspace(3)* %arg1, i8 addrspace(3)* %arg2) #0 {
  %tmp0 = bitcast i8 addrspace(3)* %arg0 to i32 addrspace(3)* addrspace(3)*
  %tmp = load volatile i32 addrspace(3)*, i32 addrspace(3)* addrspace(3)* %tmp0, align 4

  %tmp3 = load volatile i32, i32 addrspace(3)* %tmp, align 4

  %tmp4a = bitcast i8 addrspace(3)* %arg1 to i32 addrspace(3)*
  %tmp4 = load volatile i32, i32 addrspace(3)* %tmp4a, align 4

  %tmp7a = getelementptr i32, i32 addrspace(3)* %tmp, i32 8
  %tmp8 = load volatile i32, i32 addrspace(3)* %tmp7a, align 4

  %tmp9 = add i32 %tmp3, %tmp8
  %tmp10 = add i32 %tmp9, %tmp4
  %tmp14 = bitcast i32 %tmp10 to float
  ret float %tmp14
}
