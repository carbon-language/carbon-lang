; RUN: opt -debug-ir -S %s -o - | FileCheck %s

target datalayout = "e-p:64:64:64-p1:16:16:16"

define void @foo(i32 addrspace(1)*) nounwind {
  ret void
}

; Make sure the pointer size is 16

; CHECK: metadata !"0xf\00i32 addrspace(1)*\000\0016\002\000\000"
