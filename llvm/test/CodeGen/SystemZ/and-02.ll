; Test 32-bit ANDs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful NILF value.
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: nilf %r2, 1
; CHECK: br %r14
  %and = and i32 %a, 1
  ret i32 %and
}

; Check the highest 16-bit constant that must be handled by NILF.
define i32 @f2(i32 %a) {
; CHECK: f2:
; CHECK: nilf %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, 65534
  ret i32 %and
}

; ANDs of 0xffff are zero extensions from i16.
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %and = and i32 %a, 65535
  ret i32 %and
}

; Check the next value up, which must again use NILF.
define i32 @f4(i32 %a) {
; CHECK: f4:
; CHECK: nilf %r2, 65536
; CHECK: br %r14
  %and = and i32 %a, 65536
  ret i32 %and
}

; Check the lowest useful NILH value.  (LLHR is used instead of NILH of 0.)
define i32 @f5(i32 %a) {
; CHECK: f5:
; CHECK: nilh %r2, 1
; CHECK: br %r14
  %and = and i32 %a, 131071
  ret i32 %and
}

; Check the highest useful NILF value.
define i32 @f6(i32 %a) {
; CHECK: f6:
; CHECK: nilf %r2, 4294901758
; CHECK: br %r14
  %and = and i32 %a, -65538
  ret i32 %and
}

; Check the highest useful NILH value, which is one up from the above.
define i32 @f7(i32 %a) {
; CHECK: f7:
; CHECK: nilh %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, -65537
  ret i32 %and
}

; Check the low end of the NILL range, which is one up again.
define i32 @f8(i32 %a) {
; CHECK: f8:
; CHECK: nill %r2, 0
; CHECK: br %r14
  %and = and i32 %a, -65536
  ret i32 %and
}

; Check the next value up.
define i32 @f9(i32 %a) {
; CHECK: f9:
; CHECK: nill %r2, 1
; CHECK: br %r14
  %and = and i32 %a, -65535
  ret i32 %and
}

; Check the highest useful NILL value.
define i32 @f10(i32 %a) {
; CHECK: f10:
; CHECK: nill %r2, 65534
; CHECK: br %r14
  %and = and i32 %a, -2
  ret i32 %and
}
