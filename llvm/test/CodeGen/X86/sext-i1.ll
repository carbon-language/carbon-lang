; RUN: llc < %s -march=x86 | FileCheck %s
; rdar://7573216

define i32 @t1(i32 %x) nounwind readnone ssp {
entry:
; CHECK: t1:
; CHECK: cmpl $1
; CHECK: sbbl
  %0 = icmp eq i32 %x, 0
  %iftmp.0.0 = select i1 %0, i32 -1, i32 0
  ret i32 %iftmp.0.0
}

define i32 @t2(i32 %x) nounwind readnone ssp {
entry:
; CHECK: t2:
; CHECK: cmpl $1
; CHECK: sbbl
  %0 = icmp eq i32 %x, 0
  %iftmp.0.0 = sext i1 %0 to i32
  ret i32 %iftmp.0.0
}
