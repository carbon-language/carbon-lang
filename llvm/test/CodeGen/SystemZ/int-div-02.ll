; Test 32-bit unsigned division and remainder.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register division.  The result is in the second of the two registers.
define void @f1(i32 %dummy, i32 %a, i32 %b, i32 *%dest) {
; CHECK: f1:
; CHECK-NOT: %r3
; CHECK: {{llill|lhi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlr %r2, %r4
; CHECK: st %r3, 0(%r5)
; CHECK: br %r14
  %div = udiv i32 %a, %b
  store i32 %div, i32 *%dest
  ret void
}

; Test register remainder.  The result is in the first of the two registers.
define void @f2(i32 %dummy, i32 %a, i32 %b, i32 *%dest) {
; CHECK: f2:
; CHECK-NOT: %r3
; CHECK: {{llill|lhi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlr %r2, %r4
; CHECK: st %r2, 0(%r5)
; CHECK: br %r14
  %rem = urem i32 %a, %b
  store i32 %rem, i32 *%dest
  ret void
}

; Test that division and remainder use a single instruction.
define i32 @f3(i32 %dummy1, i32 %a, i32 %b) {
; CHECK: f3:
; CHECK-NOT: %r3
; CHECK: {{llill|lhi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dlr %r2, %r4
; CHECK-NOT: dlr
; CHECK: or %r2, %r3
; CHECK: br %r14
  %div = udiv i32 %a, %b
  %rem = urem i32 %a, %b
  %or = or i32 %rem, %div
  ret i32 %or
}

; Test memory division with no displacement.
define void @f4(i32 %dummy, i32 %a, i32 *%src, i32 *%dest) {
; CHECK: f4:
; CHECK-NOT: %r3
; CHECK: {{llill|lhi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dl %r2, 0(%r4)
; CHECK: st %r3, 0(%r5)
; CHECK: br %r14
  %b = load i32 *%src
  %div = udiv i32 %a, %b
  store i32 %div, i32 *%dest
  ret void
}

; Test memory remainder with no displacement.
define void @f5(i32 %dummy, i32 %a, i32 *%src, i32 *%dest) {
; CHECK: f5:
; CHECK-NOT: %r3
; CHECK: {{llill|lhi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dl %r2, 0(%r4)
; CHECK: st %r2, 0(%r5)
; CHECK: br %r14
  %b = load i32 *%src
  %rem = urem i32 %a, %b
  store i32 %rem, i32 *%dest
  ret void
}

; Test both memory division and memory remainder.
define i32 @f6(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f6:
; CHECK-NOT: %r3
; CHECK: {{llill|lhi}} %r2, 0
; CHECK-NOT: %r3
; CHECK: dl %r2, 0(%r4)
; CHECK-NOT: {{dl|dlr}}
; CHECK: or %r2, %r3
; CHECK: br %r14
  %b = load i32 *%src
  %div = udiv i32 %a, %b
  %rem = urem i32 %a, %b
  %or = or i32 %rem, %div
  ret i32 %or
}

; Check the high end of the DL range.
define i32 @f7(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f7:
; CHECK: dl %r2, 524284(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131071
  %b = load i32 *%ptr
  %rem = urem i32 %a, %b
  ret i32 %rem
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f8(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f8:
; CHECK: agfi %r4, 524288
; CHECK: dl %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131072
  %b = load i32 *%ptr
  %rem = urem i32 %a, %b
  ret i32 %rem
}

; Check the high end of the negative aligned DL range.
define i32 @f9(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f9:
; CHECK: dl %r2, -4(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -1
  %b = load i32 *%ptr
  %rem = urem i32 %a, %b
  ret i32 %rem
}

; Check the low end of the DL range.
define i32 @f10(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f10:
; CHECK: dl %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131072
  %b = load i32 *%ptr
  %rem = urem i32 %a, %b
  ret i32 %rem
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f11(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f11:
; CHECK: agfi %r4, -524292
; CHECK: dl %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131073
  %b = load i32 *%ptr
  %rem = urem i32 %a, %b
  ret i32 %rem
}

; Check that DL allows an index.
define i32 @f12(i32 %dummy, i32 %a, i64 %src, i64 %index) {
; CHECK: f12:
; CHECK: dl %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32 *%ptr
  %rem = urem i32 %a, %b
  ret i32 %rem
}
