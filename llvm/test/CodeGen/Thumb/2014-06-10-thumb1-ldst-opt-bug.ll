; RUN: llc < %s -mtriple=thumbv6m-eabi -verify-machineinstrs -o - | FileCheck %s

define void @foo(i32* %A) #0 {
entry:
; CHECK-LABEL: foo:
; CHECK: push {r7, lr}
; CHECK: ldm
; CHECK-NEXT: subs
; CHECK-NEXT: bl
  %0 = load i32* %A, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 1
  %1 = load i32* %arrayidx1, align 4
  tail call void @bar(i32* %A, i32 %0, i32 %1) #2
  ret void
}

declare void @bar(i32*, i32, i32) #1
