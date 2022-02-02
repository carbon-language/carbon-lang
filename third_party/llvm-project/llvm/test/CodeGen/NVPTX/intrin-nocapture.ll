; RUN: opt < %s -O3 -S | FileCheck %s

; Address space intrinsics were erroneously marked NoCapture, leading to bad
; optimizations (such as the store below being eliminated as dead code). This
; test makes sure we don't regress.

declare void @foo(i32 addrspace(1)*)

declare i32 addrspace(1)* @llvm.nvvm.ptr.gen.to.global.p1i32.p0i32(i32*)

; CHECK: @bar
define void @bar() {
  %t1 = alloca i32
; CHECK: call i32 addrspace(1)* @llvm.nvvm.ptr.gen.to.global.p1i32.p0i32(i32* nonnull %t1)
; CHECK-NEXT: store i32 10, i32* %t1
  %t2 = call i32 addrspace(1)* @llvm.nvvm.ptr.gen.to.global.p1i32.p0i32(i32* %t1)
  store i32 10, i32* %t1
  call void @foo(i32 addrspace(1)* %t2)
  ret void
}

