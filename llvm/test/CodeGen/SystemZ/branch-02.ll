; Test all condition-code masks that are relevant for signed integer
; comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i32 *%src, i32 %target) {
; CHECK: f1:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: j{{g?}}e .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32 *%src
  %cond = icmp eq i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i32 *%src, i32 %target) {
; CHECK: f2:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: j{{g?}}lh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32 *%src
  %cond = icmp ne i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i32 *%src, i32 %target) {
; CHECK: f3:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: j{{g?}}le .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32 *%src
  %cond = icmp sle i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i32 *%src, i32 %target) {
; CHECK: f4:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: j{{g?}}l .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32 *%src
  %cond = icmp slt i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(i32 *%src, i32 %target) {
; CHECK: f5:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: j{{g?}}h .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32 *%src
  %cond = icmp sgt i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(i32 *%src, i32 %target) {
; CHECK: f6:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: c %r3, 0(%r2)
; CHECK-NEXT: j{{g?}}he .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile i32 *%src
  %cond = icmp sge i32 %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
