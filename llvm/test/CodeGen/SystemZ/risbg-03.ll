; Test use of RISBG vs RISBGN on zEC12.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s

; On zEC12, we generally prefer RISBGN.
define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: risbgn %r2, %r3, 60, 62, 0
; CHECK: br %r14
  %anda = and i64 %a, -15
  %andb = and i64 %b, 14
  %or = or i64 %anda, %andb
  ret i64 %or
}

; But we may fall back to RISBG if we can use the condition code.
define i64 @f2(i64 %a, i64 %b, i32* %c) {
; CHECK-LABEL: f2:
; CHECK: risbg %r2, %r3, 60, 62, 0
; CHECK-NEXT: ipm
; CHECK: br %r14
  %anda = and i64 %a, -15
  %andb = and i64 %b, 14
  %or = or i64 %anda, %andb
  %cmp = icmp sgt i64 %or, 0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* %c, align 4
  ret i64 %or
}

