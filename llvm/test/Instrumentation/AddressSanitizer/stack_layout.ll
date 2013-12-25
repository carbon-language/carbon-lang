; Test the ASan's stack layout.
; More tests in tests/Transforms/Utils/ASanStackFrameLayoutTest.cpp
; RUN: opt < %s -asan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @Use(i8*)

; CHECK: private unnamed_addr constant{{.*}}3 32 10 3 XXX 64 20 3 YYY 128 30 3 ZZZ
; CHECK: private unnamed_addr constant{{.*}}3 32 5 3 AAA 64 55 3 BBB 160 555 3 CCC
; CHECK: private unnamed_addr constant{{.*}}3 256 128 3 CCC 448 128 3 BBB 608 128 3 AAA

define void @Func1() sanitize_address {
entry:
; CHECK-LABEL: Func1
; CHECK: alloca [192 x i8]
; CHECK-NOT: alloca
; CHECK: ret void
  %XXX = alloca [10 x i8], align 1
  %YYY = alloca [20 x i8], align 1
  %ZZZ = alloca [30 x i8], align 1
  ret void
}

define void @Func2() sanitize_address {
entry:
; CHECK-LABEL: Func2
; CHECK: alloca [864 x i8]
; CHECK-NOT: alloca
; CHECK: ret void
  %AAA = alloca [5 x i8], align 1
  %BBB = alloca [55 x i8], align 1
  %CCC = alloca [555 x i8], align 1
  ret void
}

; Check that we reorder vars according to alignment and handle large alignments.
define void @Func3() sanitize_address {
entry:
; CHECK-LABEL: Func3
; CHECK: alloca [768 x i8]
; CHECK-NOT: alloca
; CHECK: ret void
  %AAA = alloca [128 x i8], align 16
  %BBB = alloca [128 x i8], align 64
  %CCC = alloca [128 x i8], align 256
  ret void
}
