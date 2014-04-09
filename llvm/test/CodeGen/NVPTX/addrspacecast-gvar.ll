; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; CHECK: .visible .global .align 4 .u32 g = 42;
; CHECK: .visible .global .align 4 .u32 g2 = generic(g);
; CHECK: .visible .global .align 4 .u32 g3 = g;

@g = addrspace(1) global i32 42
@g2 = addrspace(1) global i32* addrspacecast (i32 addrspace(1)* @g to i32*)
@g3 = addrspace(1) global i32 addrspace(1)* @g
