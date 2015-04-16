; RUN: llc < %s | FileCheck %s
; This test is a sanity check to ensure statepoints are generating StackMap
; sections correctly.  This is not intended to be a rigorous test of the 
; StackMap format (see the stackmap tests for that).

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare zeroext i1 @return_i1()

define i1 @test(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test
; Do we see one spill for the local value and the store to the
; alloca?
; CHECK: subq	$24, %rsp
; CHECK: movq	$0, 8(%rsp)
; CHECK: movq	%rdi, (%rsp)
; CHECK: callq return_i1
; CHECK: addq	$24, %rsp
; CHECK: retq
entry:
  %metadata1 = alloca i32 addrspace(1)*, i32 2, align 8
  store i32 addrspace(1)* null, i32 addrspace(1)** %metadata1
  %safepoint_token = tail call i32 (i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 2, i32 addrspace(1)* %ptr, i32 addrspace(1)* null, i32 addrspace(1)* %ptr, i32 addrspace(1)* null)
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(i32 %safepoint_token)
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 6, i32 6)
  %b = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 7, i32 7)
; 
  ret i1 %call1
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()*, i32, i32, ...)
declare i1 @llvm.experimental.gc.result.i1(i32)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32) #3


; CHECK-LABEL: .section .llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 1
; Num LargeConstants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 1

; Functions and stack size
; CHECK-NEXT:   .quad test
; CHECK-NEXT:   .quad 24

; Large Constants
; Statepoint ID only
; CHECK: .quad	2882400000

; Callsites
; Constant arguments
; CHECK: .long	.Ltmp1-test
; CHECK: .short	0
; CHECK: .short	8
; SmallConstant (0)
; CHECK: .byte	4
; CHECK: .byte	8
; CHECK: .short	0
; CHECK: .long	0
; SmallConstant (2)
; CHECK: .byte	4
; CHECK: .byte	8
; CHECK: .short	0
; CHECK: .long	2
; Direct Spill Slot [RSP+0]
; CHECK: .byte	2
; CHECK: .byte	8
; CHECK: .short	7
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK: .byte	8
; CHECK: .short	0
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK: .byte	8
; CHECK: .short	0
; CHECK: .long	0
; SmallConstant  (0)
; CHECK: .byte	4
; CHECK: .byte	8
; CHECK: .short	0
; CHECK: .long	0
; Direct Spill Slot [RSP+0]
; CHECK: .byte	2
; CHECK: .byte	8
; CHECK: .short	7
; CHECK: .long	0
; Direct Spill Slot [RSP+0]
; CHECK: .byte	2
; CHECK: .byte	8
; CHECK: .short	7
; CHECK: .long	0

; No Padding or LiveOuts
; CHECK: .short	0
; CHECK: .short	0
; CHECK: .align	8


