; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

; Test that global aliases are allowed to be constant addrspacecast

@i = internal addrspace(1) global i8 42
@ia = internal alias addrspacecast (i8 addrspace(1)* @i to i8 addrspace(2)* addrspace(3)*)
; CHECK: @ia = internal alias addrspacecast (i8 addrspace(2)* addrspace(1)* bitcast (i8 addrspace(1)* @i to i8 addrspace(2)* addrspace(1)*) to i8 addrspace(2)* addrspace(3)*)
