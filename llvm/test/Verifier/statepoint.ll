; RUN: opt -S %s -verify | FileCheck %s

declare void @use(...)
declare i8 addrspace(1)* @llvm.gc.relocate.p1i8(i32, i32, i32)
declare i32 @llvm.statepoint.p0f_isVoidf(void ()*, i32, i32, ...)

;; Basic usage
define i8 addrspace(1)* @test1(i8 addrspace(1)* %arg) {
entry:
  %cast = bitcast i8 addrspace(1)* %arg to i64 addrspace(1)*
  %safepoint_token = call i32 (void ()*, i32, i32, ...)* @llvm.statepoint.p0f_isVoidf(void ()* undef, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg, i8 addrspace(1)* %arg)
  %reloc = call i8 addrspace(1)* @llvm.gc.relocate.p1i8(i32 %safepoint_token, i32 9, i32 10)
  ;; It is perfectly legal to relocate the same value multiple times...
  %reloc2 = call i8 addrspace(1)* @llvm.gc.relocate.p1i8(i32 %safepoint_token, i32 9, i32 10)
  %reloc3 = call i8 addrspace(1)* @llvm.gc.relocate.p1i8(i32 %safepoint_token, i32 10, i32 9)
  ret i8 addrspace(1)* %reloc
; CHECK-LABEL: test1
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK: gc.relocate
; CHECK: gc.relocate
; CHECK: ret i8 addrspace(1)* %reloc
}

; This test catches two cases where the verifier was too strict:
; 1) A base doesn't need to be relocated if it's never used again
; 2) A value can be replaced by one which is known equal.  This
; means a potentially derived pointer can be known base and that
; we can't check that derived pointer are never bases.
define void @test2(i8 addrspace(1)* %arg, i64 addrspace(1)* %arg2) {
entry:
  %cast = bitcast i8 addrspace(1)* %arg to i64 addrspace(1)*
  %c = icmp eq i64 addrspace(1)* %cast,  %arg2
  br i1 %c, label %equal, label %notequal

notequal:
  ret void

equal:
%safepoint_token = call i32 (void ()*, i32, i32, ...)* @llvm.statepoint.p0f_isVoidf(void ()* undef, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg, i8 addrspace(1)* %arg)
  %reloc = call i8 addrspace(1)* @llvm.gc.relocate.p1i8(i32 %safepoint_token, i32 9, i32 10)
  call void undef(i8 addrspace(1)* %reloc)
  ret void
; CHECK-LABEL: test2
; CHECK-LABEL: equal
; CHECK: statepoint
; CHECK-NEXT: %reloc = call 
; CHECK-NEXT: call
; CHECK-NEXT: ret voi
}
