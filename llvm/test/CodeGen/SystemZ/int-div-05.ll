; Testg 64-bit unsigned division and remainder.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Testg register division.  The result is in the second of the two registers.
define void @f1(i64 %dummy, i64 %a, i64 %b, i64 *%dest) {
; CHECK: f1:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlgr %r2, %r4
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %div = udiv i64 %a, %b
  store i64 %div, i64 *%dest
  ret void
}

; Testg register remainder.  The result is in the first of the two registers.
define void @f2(i64 %dummy, i64 %a, i64 %b, i64 *%dest) {
; CHECK: f2:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlgr %r2, %r4
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %rem = urem i64 %a, %b
  store i64 %rem, i64 *%dest
  ret void
}

; Testg that division and remainder use a single instruction.
define i64 @f3(i64 %dummy1, i64 %a, i64 %b) {
; CHECK: f3:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlgr %r2, %r4
; CHECK-NOT: dlgr
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %div = udiv i64 %a, %b
  %rem = urem i64 %a, %b
  %or = or i64 %rem, %div
  ret i64 %or
}

; Testg memory division with no displacement.
define void @f4(i64 %dummy, i64 %a, i64 *%src, i64 *%dest) {
; CHECK: f4:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlg %r2, 0(%r4)
; CHECK: stg %r3, 0(%r5)
; CHECK: br %r14
  %b = load i64 *%src
  %div = udiv i64 %a, %b
  store i64 %div, i64 *%dest
  ret void
}

; Testg memory remainder with no displacement.
define void @f5(i64 %dummy, i64 %a, i64 *%src, i64 *%dest) {
; CHECK: f5:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlg %r2, 0(%r4)
; CHECK: stg %r2, 0(%r5)
; CHECK: br %r14
  %b = load i64 *%src
  %rem = urem i64 %a, %b
  store i64 %rem, i64 *%dest
  ret void
}

; Testg both memory division and memory remainder.
define i64 @f6(i64 %dummy, i64 %a, i64 *%src) {
; CHECK: f6:
; CHECK-NOT: %r3
; CHECK: {{llill|lghi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlg %r2, 0(%r4)
; CHECK-NOT: {{dlg|dlgr}}
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %b = load i64 *%src
  %div = udiv i64 %a, %b
  %rem = urem i64 %a, %b
  %or = or i64 %rem, %div
  ret i64 %or
}

; Check the high end of the DLG range.
define i64 @f7(i64 %dummy, i64 %a, i64 *%src) {
; CHECK: f7:
; CHECK: dlg %r2, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65535
  %b = load i64 *%ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f8(i64 %dummy, i64 %a, i64 *%src) {
; CHECK: f8:
; CHECK: agfi %r4, 524288
; CHECK: dlg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65536
  %b = load i64 *%ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the high end of the negative aligned DLG range.
define i64 @f9(i64 %dummy, i64 %a, i64 *%src) {
; CHECK: f9:
; CHECK: dlg %r2, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -1
  %b = load i64 *%ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the low end of the DLG range.
define i64 @f10(i64 %dummy, i64 %a, i64 *%src) {
; CHECK: f10:
; CHECK: dlg %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65536
  %b = load i64 *%ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f11(i64 %dummy, i64 %a, i64 *%src) {
; CHECK: f11:
; CHECK: agfi %r4, -524296
; CHECK: dlg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65537
  %b = load i64 *%ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}

; Check that DLG allows an index.
define i64 @f12(i64 %dummy, i64 %a, i64 %src, i64 %index) {
; CHECK: f12:
; CHECK: dlg %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64 *%ptr
  %rem = urem i64 %a, %b
  ret i64 %rem
}
