; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:4"
target triple = "x86_64-unknown-linux-gnu"

define i8 addrspace(4)* @f_0() {
; CHECK-LABEL: @f_0(
; CHECK: ret i8 addrspace(4)* getelementptr (i8, i8 addrspace(4)* null, i64 50)
  %result = getelementptr i8, i8 addrspace(4)* null, i64 50
  ret i8 addrspace(4)* %result
}

define i8 addrspace(3)* @f_1() {
; inttoptr is fine here since addrspace(3) is integral.

; CHECK-LABEL: @f_1(
; CHECK: ret i8 addrspace(3)* inttoptr (i64 50 to i8 addrspace(3)*)
  %result = getelementptr i8, i8 addrspace(3)* null, i64 50
  ret i8 addrspace(3)* %result
}
