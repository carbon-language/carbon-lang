; Test all condition-code masks that are relevant for CGRJ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

define void @f1(i64 %target) {
; CHECK: f1:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrje %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp eq i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i64 %target) {
; CHECK: f2:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjlh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp ne i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i64 %target) {
; CHECK: f3:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjle %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp sle i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i64 %target) {
; CHECK: f4:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjl %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp slt i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(i64 %target) {
; CHECK: f5:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp sgt i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(i64 %target) {
; CHECK: f6:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: cgrjhe %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i64 @foo()
  %cond = icmp sge i64 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
