; RUN: opt -instcombine -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-n8:16:32:64"

define i32* @pointer_to_addrspace_pointer(i32 addrspace(1)** %x) nounwind {
; CHECK-LABEL: @pointer_to_addrspace_pointer(
; CHECK: load
; CHECK: addrspacecast
  %y = bitcast i32 addrspace(1)** %x to i32**
  %z = load i32** %y
  ret i32* %z
}

