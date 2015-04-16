; Test that we can correctly handle vectors of pointers in statepoint 
; rewriting.  Currently, we scalarize, but that's an implementation detail.
; RUN: opt %s -rewrite-statepoints-for-gc -S | FileCheck  %s

; A non-vector relocation for comparison
define i64 addrspace(1)* @test(i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: test
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: ret i64 addrspace(1)* %obj.relocated
entry:
  %safepoint_token = call i32 (void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @do_safepoint, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}

; A base vector from a argument
define <2 x i64 addrspace(1)*> @test2(<2 x i64 addrspace(1)*> %obj) gc "statepoint-example" {
; CHECK-LABEL: test2
; CHECK: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %5
entry:
  %safepoint_token = call i32 (void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @do_safepoint, i32 0, i32 0, i32 0)
  ret <2 x i64 addrspace(1)*> %obj
}

; A base vector from a load
define <2 x i64 addrspace(1)*> @test3(<2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test3
; CHECK: load
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %5
entry:
  %obj = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  %safepoint_token = call i32 (void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @do_safepoint, i32 0, i32 0, i32 0)
  ret <2 x i64 addrspace(1)*> %obj
}

declare i32 @fake_personality_function()

; When a statepoint is an invoke rather than a call
define <2 x i64 addrspace(1)*> @test4(<2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test4
; CHECK: load
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
entry:
  %obj = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  invoke i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @do_safepoint, i32 0, i32 0, i32 0)
          to label %normal_return unwind label %exceptional_return

; CHECK-LABEL: normal_return:
; CHECK: gc.relocate
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %6
normal_return:                                    ; preds = %entry
  ret <2 x i64 addrspace(1)*> %obj

; CHECK-LABEL: exceptional_return:
; CHECK: gc.relocate
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %10
exceptional_return:                               ; preds = %entry
  %landing_pad4 = landingpad { i8*, i32 } personality i32 ()* @fake_personality_function
          cleanup
  ret <2 x i64 addrspace(1)*> %obj
}

declare void @do_safepoint()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()*, i32, i32, ...)
