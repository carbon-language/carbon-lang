; Test all condition-code masks that are relevant for floating-point
; comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(float *%src, float %target) {
; CHECK: f1:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}e .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp oeq float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(float *%src, float %target) {
; CHECK: f2:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}lh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp one float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(float *%src, float %target) {
; CHECK: f3:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}le .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ole float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(float *%src, float %target) {
; CHECK: f4:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}l .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp olt float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(float *%src, float %target) {
; CHECK: f5:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}h .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ogt float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(float *%src, float %target) {
; CHECK: f6:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}he .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp oge float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f7(float *%src, float %target) {
; CHECK: f7:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}nlh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ueq float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f8(float *%src, float %target) {
; CHECK: f8:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}ne .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp une float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f9(float *%src, float %target) {
; CHECK: f9:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}nh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ule float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f10(float *%src, float %target) {
; CHECK: f10:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}nhe .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ult float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f11(float *%src, float %target) {
; CHECK: f11:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}nle .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ugt float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f12(float *%src, float %target) {
; CHECK: f12:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}nl .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp uge float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; "jno" == "jump if no overflow", which corresponds to "jump if ordered"
; rather than "jump if not ordered" after a floating-point comparison.
define void @f13(float *%src, float %target) {
; CHECK: f13:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}no .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp ord float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; "jo" == "jump if overflow", which corresponds to "jump if not ordered"
; rather than "jump if ordered" after a floating-point comparison.
define void @f14(float *%src, float %target) {
; CHECK: f14:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: j{{g?}}o .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float *%src
  %cond = fcmp uno float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
