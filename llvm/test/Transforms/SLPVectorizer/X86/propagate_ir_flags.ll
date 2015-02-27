; RUN: opt < %s -basicaa -slp-vectorizer -S | FileCheck %s

; Check propagation of optional IR flags (PR20802). For a flag to
; propagate from scalar instructions to their vector replacement,
; *all* scalar instructions must have the flag.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; CHECK-LABEL: @exact(
; CHECK: lshr exact <4 x i32>
define void @exact(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = lshr exact i32 %load1, 1
  %op2 = lshr exact i32 %load2, 1
  %op3 = lshr exact i32 %load3, 1
  %op4 = lshr exact i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}

; CHECK-LABEL: @not_exact(
; CHECK: lshr <4 x i32>
define void @not_exact(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = lshr exact i32 %load1, 1
  %op2 = lshr i32 %load2, 1
  %op3 = lshr exact i32 %load3, 1
  %op4 = lshr exact i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}

; CHECK-LABEL: @nsw(
; CHECK: add nsw <4 x i32>
define void @nsw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add nsw i32 %load1, 1
  %op2 = add nsw i32 %load2, 1
  %op3 = add nsw i32 %load3, 1
  %op4 = add nsw i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}

; CHECK-LABEL: @not_nsw(
; CHECK: add <4 x i32>
define void @not_nsw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add nsw i32 %load1, 1
  %op2 = add nsw i32 %load2, 1
  %op3 = add nsw i32 %load3, 1
  %op4 = add i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}

; CHECK-LABEL: @nuw(
; CHECK: add nuw <4 x i32>
define void @nuw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add nuw i32 %load1, 1
  %op2 = add nuw i32 %load2, 1
  %op3 = add nuw i32 %load3, 1
  %op4 = add nuw i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @not_nuw(
; CHECK: add <4 x i32>
define void @not_nuw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add nuw i32 %load1, 1
  %op2 = add i32 %load2, 1
  %op3 = add i32 %load3, 1
  %op4 = add nuw i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @nnan(
; CHECK: fadd nnan <4 x float>
define void @nnan(float* %x) {
  %idx1 = getelementptr inbounds float, float* %x, i64 0
  %idx2 = getelementptr inbounds float, float* %x, i64 1
  %idx3 = getelementptr inbounds float, float* %x, i64 2
  %idx4 = getelementptr inbounds float, float* %x, i64 3

  %load1 = load float* %idx1, align 4
  %load2 = load float* %idx2, align 4
  %load3 = load float* %idx3, align 4
  %load4 = load float* %idx4, align 4

  %op1 = fadd fast nnan float %load1, 1.0
  %op2 = fadd nnan ninf float %load2, 1.0
  %op3 = fadd nsz nnan float %load3, 1.0
  %op4 = fadd arcp nnan float %load4, 1.0

  store float %op1, float* %idx1, align 4
  store float %op2, float* %idx2, align 4
  store float %op3, float* %idx3, align 4
  store float %op4, float* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @not_nnan(
; CHECK: fadd <4 x float>
define void @not_nnan(float* %x) {
  %idx1 = getelementptr inbounds float, float* %x, i64 0
  %idx2 = getelementptr inbounds float, float* %x, i64 1
  %idx3 = getelementptr inbounds float, float* %x, i64 2
  %idx4 = getelementptr inbounds float, float* %x, i64 3

  %load1 = load float* %idx1, align 4
  %load2 = load float* %idx2, align 4
  %load3 = load float* %idx3, align 4
  %load4 = load float* %idx4, align 4

  %op1 = fadd nnan float %load1, 1.0
  %op2 = fadd ninf float %load2, 1.0
  %op3 = fadd nsz float %load3, 1.0
  %op4 = fadd arcp float %load4, 1.0

  store float %op1, float* %idx1, align 4
  store float %op2, float* %idx2, align 4
  store float %op3, float* %idx3, align 4
  store float %op4, float* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @only_fast(
; CHECK: fadd fast <4 x float>
define void @only_fast(float* %x) {
  %idx1 = getelementptr inbounds float, float* %x, i64 0
  %idx2 = getelementptr inbounds float, float* %x, i64 1
  %idx3 = getelementptr inbounds float, float* %x, i64 2
  %idx4 = getelementptr inbounds float, float* %x, i64 3

  %load1 = load float* %idx1, align 4
  %load2 = load float* %idx2, align 4
  %load3 = load float* %idx3, align 4
  %load4 = load float* %idx4, align 4

  %op1 = fadd fast nnan float %load1, 1.0
  %op2 = fadd fast nnan ninf float %load2, 1.0
  %op3 = fadd fast nsz nnan float %load3, 1.0
  %op4 = fadd arcp nnan fast float %load4, 1.0

  store float %op1, float* %idx1, align 4
  store float %op2, float* %idx2, align 4
  store float %op3, float* %idx3, align 4
  store float %op4, float* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @only_arcp(
; CHECK: fadd arcp <4 x float>
define void @only_arcp(float* %x) {
  %idx1 = getelementptr inbounds float, float* %x, i64 0
  %idx2 = getelementptr inbounds float, float* %x, i64 1
  %idx3 = getelementptr inbounds float, float* %x, i64 2
  %idx4 = getelementptr inbounds float, float* %x, i64 3

  %load1 = load float* %idx1, align 4
  %load2 = load float* %idx2, align 4
  %load3 = load float* %idx3, align 4
  %load4 = load float* %idx4, align 4

  %op1 = fadd fast float %load1, 1.0
  %op2 = fadd fast float %load2, 1.0
  %op3 = fadd fast float %load3, 1.0
  %op4 = fadd arcp float %load4, 1.0

  store float %op1, float* %idx1, align 4
  store float %op2, float* %idx2, align 4
  store float %op3, float* %idx3, align 4
  store float %op4, float* %idx4, align 4

  ret void
}

; CHECK-LABEL: @addsub_all_nsw
; CHECK: add nsw <4 x i32>
; CHECK: sub nsw <4 x i32>
define void @addsub_all_nsw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add nsw i32 %load1, 1
  %op2 = sub nsw i32 %load2, 1
  %op3 = add nsw i32 %load3, 1
  %op4 = sub nsw i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @addsub_some_nsw
; CHECK: add nsw <4 x i32>
; CHECK: sub <4 x i32>
define void @addsub_some_nsw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add nsw i32 %load1, 1
  %op2 = sub nsw i32 %load2, 1
  %op3 = add nsw i32 %load3, 1
  %op4 = sub i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}
 
; CHECK-LABEL: @addsub_no_nsw
; CHECK: add <4 x i32>
; CHECK: sub <4 x i32>
define void @addsub_no_nsw(i32* %x) {
  %idx1 = getelementptr inbounds i32, i32* %x, i64 0
  %idx2 = getelementptr inbounds i32, i32* %x, i64 1
  %idx3 = getelementptr inbounds i32, i32* %x, i64 2
  %idx4 = getelementptr inbounds i32, i32* %x, i64 3

  %load1 = load i32* %idx1, align 4
  %load2 = load i32* %idx2, align 4
  %load3 = load i32* %idx3, align 4
  %load4 = load i32* %idx4, align 4

  %op1 = add i32 %load1, 1
  %op2 = sub nsw i32 %load2, 1
  %op3 = add nsw i32 %load3, 1
  %op4 = sub i32 %load4, 1

  store i32 %op1, i32* %idx1, align 4
  store i32 %op2, i32* %idx2, align 4
  store i32 %op3, i32* %idx3, align 4
  store i32 %op4, i32* %idx4, align 4

  ret void
}
 
