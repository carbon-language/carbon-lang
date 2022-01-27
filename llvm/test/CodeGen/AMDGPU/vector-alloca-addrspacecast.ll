; RUN: opt -S -mtriple=amdgcn-- -data-layout=A5 -amdgpu-promote-alloca -sroa -instcombine < %s | FileCheck -check-prefix=OPT %s

; Should give up promoting alloca to vector with an addrspacecast.

; OPT-LABEL: @vector_addrspacecast(
; OPT: alloca [3 x i32]
; OPT: store i32 0, i32 addrspace(5)* %a0, align 4
; OPT: store i32 1, i32 addrspace(5)* %a1, align 4
; OPT: store i32 2, i32 addrspace(5)* %a2, align 4
; OPT: %tmp = getelementptr [3 x i32], [3 x i32] addrspace(5)* %alloca, i64 0, i64 %index
; OPT: %ac = addrspacecast i32 addrspace(5)* %tmp to i32*
; OPT: %data = load i32, i32* %ac, align 4
define amdgpu_kernel void @vector_addrspacecast(i32 addrspace(1)* %out, i64 %index) {
entry:
  %alloca = alloca [3 x i32], addrspace(5)
  %a0 = getelementptr [3 x i32], [3 x i32] addrspace(5)* %alloca, i32 0, i32 0
  %a1 = getelementptr [3 x i32], [3 x i32] addrspace(5)* %alloca, i32 0, i32 1
  %a2 = getelementptr [3 x i32], [3 x i32] addrspace(5)* %alloca, i32 0, i32 2
  store i32 0, i32 addrspace(5)* %a0
  store i32 1, i32 addrspace(5)* %a1
  store i32 2, i32 addrspace(5)* %a2
  %tmp = getelementptr [3 x i32], [3 x i32] addrspace(5)* %alloca, i64 0, i64 %index
  %ac = addrspacecast i32 addrspace(5)* %tmp to i32 *
  %data = load i32, i32 * %ac
  store i32 %data, i32 addrspace(1)* %out
  ret void
}
