; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Test that global aliases are allowed to be constant addrspacecast

@i = internal addrspace(1) global i8 42
@ia = alias internal addrspace(2) i8 addrspace(3)*, i8 addrspace(1)* @i
; CHECK: @ia = alias internal addrspace(2) i8 addrspace(3)*, i8 addrspace(1)* @i
