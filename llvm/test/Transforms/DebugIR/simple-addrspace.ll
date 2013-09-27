; RUN: opt -debug-ir -S %s -o - | FileCheck %s

target datalayout = "e-p:64:64:64-p1:16:16:16"

define void @foo(i32 addrspace(1)*) nounwind {
  ret void
}

; Make sure the pointer size is 16

; CHECK: metadata !"i32 addrspace(1)*", i32 0, i64 16, i64 2, i64 0, i32 0


