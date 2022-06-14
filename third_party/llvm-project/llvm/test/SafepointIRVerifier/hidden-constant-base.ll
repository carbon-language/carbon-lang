; RUN: opt -safepoint-ir-verifier-print-only -passes='verify<safepoint-ir>' -S %s 2>&1 | FileCheck %s

; CHECK-LABEL: Verifying gc pointers in function: test
; CHECK: No illegal uses found by SafepointIRVerifier in: test_gc_relocate
define i8 addrspace(1)* @test_gc_relocate(i1 %arg) gc "statepoint-example" {
bb:
  br label %bb2

bb2:                                              ; preds = %bb8, %bb
  %tmp = phi i8 addrspace(1)* [ %tmp5.relocated, %bb8 ], [ null, %bb ]
  %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* elementtype(void ()) @widget, i32 0, i32 0, i32 0, i32 0) [ "deopt"() ]
  br label %bb4

bb4:                                              ; preds = %bb8, %bb2
  %tmp5 = phi i8 addrspace(1)* [ %tmp5.relocated, %bb8 ], [ %tmp, %bb2 ]
  %statepoint_token1 = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 2882400000, i32 0, i1 ()* elementtype(i1 ()) @baz, i32 0, i32 0, i32 0, i32 0) [ "deopt"(i8 addrspace(1)* %tmp5), "gc-live"(i8 addrspace(1)* %tmp5) ]
  %tmp62 = call i1 @llvm.experimental.gc.result.i1(token %statepoint_token1)
  %tmp5.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token1, i32 0, i32 0) ; (%tmp5, %tmp5)
  br i1 %tmp62, label %bb8, label %bb6

bb6:                                              ; preds = %bb4
  ret i8 addrspace(1)* null

bb8:                                              ; preds = %bb4
  br i1 %arg, label %bb4, label %bb2
}

; CHECK-LABEL: Verifying gc pointers in function: test
; CHECK: No illegal uses found by SafepointIRVerifier in: test_freeze_inst
define void @test_freeze_inst(i32* %ptr) gc "statepoint-example" {
entry:
  br label %loop

loop:
  %vptr = phi i8 addrspace(1)* [ %fptr, %loop ], [ undef, %entry ]
  %fptr = freeze i8 addrspace(1)* %vptr
  %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0) [ "deopt"(i8 addrspace(1)* %fptr), "gc-live"(i8 addrspace(1)* %fptr) ]
  %fptr.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 0, i32 0) ; (%fptr, %fptr)
  br label %loop
}

declare void @widget()
declare i1 @baz()

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 immarg, i32 immarg, void ()*, i32 immarg, i32 immarg, ...)
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64 immarg, i32 immarg, i1 ()*, i32 immarg, i32 immarg, ...)
declare i1 @llvm.experimental.gc.result.i1(token)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32 immarg, i32 immarg)
