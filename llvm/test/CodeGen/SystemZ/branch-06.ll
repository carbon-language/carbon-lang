; Test all condition-code masks that are relevant for CRJ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

define void @f1(i32 %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crje %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp eq i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(i32 %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjlh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp ne i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(i32 %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjle %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp sle i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(i32 %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjl %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp slt i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(i32 %target) {
; CHECK-LABEL: f5:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjh %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp sgt i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(i32 %target) {
; CHECK-LABEL: f6:
; CHECK: .cfi_def_cfa_offset
; CHECK: .L[[LABEL:.*]]:
; CHECK: crjhe %r2, {{%r[0-9]+}}, .L[[LABEL]]
  br label %loop
loop:
  %val = call i32 @foo()
  %cond = icmp sge i32 %val, %target
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
