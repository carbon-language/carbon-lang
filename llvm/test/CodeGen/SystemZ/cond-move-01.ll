; Test LOCR and LOCGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Test LOCR.
define i32 @f1(i32 %a, i32 %b, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK: clfi %r4, 42
; CHECK: locrnl %r2, %r3
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  ret i32 %res
}

; Test LOCGR.
define i64 @f2(i64 %a, i64 %b, i64 %limit) {
; CHECK-LABEL: f2:
; CHECK: clgfi %r4, 42
; CHECK: locgrnl %r2, %r3
; CHECK: br %r14
  %cond = icmp ult i64 %limit, 42
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
