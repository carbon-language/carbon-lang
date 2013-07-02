; Test 32-bit signed division and remainder.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Test register division.  The result is in the second of the two registers.
define void @f1(i32 *%dest, i32 %a, i32 %b) {
; CHECK: f1:
; CHECK: lgfr %r1, %r3
; CHECK: dsgfr %r0, %r4
; CHECK: st %r1, 0(%r2)
; CHECK: br %r14
  %div = sdiv i32 %a, %b
  store i32 %div, i32 *%dest
  ret void
}

; Test register remainder.  The result is in the first of the two registers.
define void @f2(i32 *%dest, i32 %a, i32 %b) {
; CHECK: f2:
; CHECK: lgfr %r1, %r3
; CHECK: dsgfr %r0, %r4
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %rem = srem i32 %a, %b
  store i32 %rem, i32 *%dest
  ret void
}

; Test that division and remainder use a single instruction.
define i32 @f3(i32 %dummy, i32 %a, i32 %b) {
; CHECK: f3:
; CHECK-NOT: %r2
; CHECK: lgfr %r3, %r3
; CHECK-NOT: %r2
; CHECK: dsgfr %r2, %r4
; CHECK-NOT: dsgfr
; CHECK: or %r2, %r3
; CHECK: br %r14
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
  %or = or i32 %rem, %div
  ret i32 %or
}

; Check that the sign extension of the dividend is elided when the argument
; is already sign-extended.
define i32 @f4(i32 %dummy, i32 signext %a, i32 %b) {
; CHECK: f4:
; CHECK-NOT: {{%r[234]}}
; CHECK: dsgfr %r2, %r4
; CHECK-NOT: dsgfr
; CHECK: or %r2, %r3
; CHECK: br %r14
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
  %or = or i32 %rem, %div
  ret i32 %or
}

; Test that memory dividends are loaded using sign extension (LGF).
define i32 @f5(i32 %dummy, i32 *%src, i32 %b) {
; CHECK: f5:
; CHECK-NOT: %r2
; CHECK: lgf %r3, 0(%r3)
; CHECK-NOT: %r2
; CHECK: dsgfr %r2, %r4
; CHECK-NOT: dsgfr
; CHECK: or %r2, %r3
; CHECK: br %r14
  %a = load i32 *%src
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
  %or = or i32 %rem, %div
  ret i32 %or
}

; Test memory division with no displacement.
define void @f6(i32 *%dest, i32 %a, i32 *%src) {
; CHECK: f6:
; CHECK: lgfr %r1, %r3
; CHECK: dsgf %r0, 0(%r4)
; CHECK: st %r1, 0(%r2)
; CHECK: br %r14
  %b = load i32 *%src
  %div = sdiv i32 %a, %b
  store i32 %div, i32 *%dest
  ret void
}

; Test memory remainder with no displacement.
define void @f7(i32 *%dest, i32 %a, i32 *%src) {
; CHECK: f7:
; CHECK: lgfr %r1, %r3
; CHECK: dsgf %r0, 0(%r4)
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %b = load i32 *%src
  %rem = srem i32 %a, %b
  store i32 %rem, i32 *%dest
  ret void
}

; Test both memory division and memory remainder.
define i32 @f8(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f8:
; CHECK-NOT: %r2
; CHECK: lgfr %r3, %r3
; CHECK-NOT: %r2
; CHECK: dsgf %r2, 0(%r4)
; CHECK-NOT: {{dsgf|dsgfr}}
; CHECK: or %r2, %r3
; CHECK: br %r14
  %b = load i32 *%src
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
  %or = or i32 %rem, %div
  ret i32 %or
}

; Check the high end of the DSGF range.
define i32 @f9(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f9:
; CHECK: dsgf %r2, 524284(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131071
  %b = load i32 *%ptr
  %rem = srem i32 %a, %b
  ret i32 %rem
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f10(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f10:
; CHECK: agfi %r4, 524288
; CHECK: dsgf %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131072
  %b = load i32 *%ptr
  %rem = srem i32 %a, %b
  ret i32 %rem
}

; Check the high end of the negative aligned DSGF range.
define i32 @f11(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f11:
; CHECK: dsgf %r2, -4(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -1
  %b = load i32 *%ptr
  %rem = srem i32 %a, %b
  ret i32 %rem
}

; Check the low end of the DSGF range.
define i32 @f12(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f12:
; CHECK: dsgf %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131072
  %b = load i32 *%ptr
  %rem = srem i32 %a, %b
  ret i32 %rem
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f13(i32 %dummy, i32 %a, i32 *%src) {
; CHECK: f13:
; CHECK: agfi %r4, -524292
; CHECK: dsgf %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131073
  %b = load i32 *%ptr
  %rem = srem i32 %a, %b
  ret i32 %rem
}

; Check that DSGF allows an index.
define i32 @f14(i32 %dummy, i32 %a, i64 %src, i64 %index) {
; CHECK: f14:
; CHECK: dsgf %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32 *%ptr
  %rem = srem i32 %a, %b
  ret i32 %rem
}

; Make sure that we still use DSGFR rather than DSGR in cases where
; a load and division cannot be combined.
define void @f15(i32 *%dest, i32 *%src) {
; CHECK: f15:
; CHECK: l [[B:%r[0-9]+]], 0(%r3)
; CHECK: brasl %r14, foo@PLT
; CHECK: lgfr %r1, %r2
; CHECK: dsgfr %r0, [[B]]
; CHECK: br %r14
  %b = load i32 *%src
  %a = call i32 @foo()
  %div = sdiv i32 %a, %b
  store i32 %div, i32 *%dest
  ret void
}
