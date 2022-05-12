; RUN: llc  -verify-machineinstrs < %s | FileCheck %s

; Without removal of duplicate entries, the size is 62 lines
;      CHECK:	.section	.llvm_stackmaps,{{.*$}}
; CHECK-NEXT:{{(.+$[[:space:]]){48}[[:space:]]}}
;  CHECK-NOT:{{.|[[:space:]]}}

target triple = "x86_64-pc-linux-gnu"

declare void @func()

define i1 @test1(i32 addrspace(1)* %arg) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %arg)]
  %reloc1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %reloc2 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %cmp1 = icmp eq i32 addrspace(1)* %reloc1, null
  %cmp2 = icmp eq i32 addrspace(1)* %reloc2, null
  %cmp = and i1 %cmp1, %cmp2
  ret i1 %cmp
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
