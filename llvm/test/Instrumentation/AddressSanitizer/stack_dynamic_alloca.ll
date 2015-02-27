; RUN: opt < %s -asan -asan-module -asan-stack-dynamic-alloca \
; RUN:       -asan-use-after-return -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @Func1() sanitize_address {
entry:
; CHECK-LABEL: Func1

; CHECK: entry:
; CHECK: load i32* @__asan_option_detect_stack_use_after_return

; CHECK: <label>:[[UAR_ENABLED_BB:[0-9]+]]
; CHECK: [[FAKE_STACK_RT:%[0-9]+]] = call i64 @__asan_stack_malloc_

; CHECK: <label>:[[FAKE_STACK_BB:[0-9]+]]
; CHECK: [[FAKE_STACK:%[0-9]+]] = phi i64 [ 0, %entry ], [ [[FAKE_STACK_RT]], %[[UAR_ENABLED_BB]] ]
; CHECK: icmp eq i64 [[FAKE_STACK]], 0

; CHECK: <label>:[[NO_FAKE_STACK_BB:[0-9]+]]
; CHECK: %MyAlloca = alloca i8, i64
; CHECK: [[ALLOCA:%[0-9]+]] = ptrtoint i8* %MyAlloca

; CHECK: phi i64 [ [[FAKE_STACK]], %[[FAKE_STACK_BB]] ], [ [[ALLOCA]], %[[NO_FAKE_STACK_BB]] ]

; CHECK: ret void

  %XXX = alloca [20 x i8], align 1
  %arr.ptr = bitcast [20 x i8]* %XXX to i8*
  store volatile i8 0, i8* %arr.ptr
  ret void
}

; Test that dynamic alloca is not used for functions with inline assembly.
define void @Func2() sanitize_address {
entry:
; CHECK-LABEL: Func2
; CHECK: alloca [96 x i8]
; CHECK: ret void

  %XXX = alloca [20 x i8], align 1
  %arr.ptr = bitcast [20 x i8]* %XXX to i8*
  store volatile i8 0, i8* %arr.ptr
  call void asm sideeffect "mov %%rbx, %%rcx", "~{dirflag},~{fpsr},~{flags}"() nounwind
  ret void
}
