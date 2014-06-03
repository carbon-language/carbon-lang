; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Test that global aliases are allowed to be constant addrspacecast

@i = internal addrspace(1) global i8 42
@ia = alias internal addrspacecast (i8 addrspace(1)* @i to i8 addrspace(2)* addrspace(3)*)
; CHECK: @ia =  alias internal addrspacecast (i8 addrspace(1)* @i to i8 addrspace(2)* addrspace(3)*)
