; Test 32-bit ANDs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; ANDs with 1 should use RISBG
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: risbg %r2, %r2, 63, 191, 0
; CHECK: br %r14
  %and = and i32 %a, 1
  ret i32 %and
}

; ...same for 2.
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: risbg %r2, %r2, 62, 190, 0
; CHECK: br %r14
  %and = and i32 %a, 2
  ret i32 %and
}

; ...and 3.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: risbg %r2, %r2, 62, 191, 0
; CHECK: br %r14
  %and = and i32 %a, 3
  ret i32 %and
}

; ...and 4.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: risbg %r2, %r2, 61, 189, 0
; CHECK: br %r14
  %and = and i32 %a, 4
  ret i32 %and
}

; Check the lowest useful NILF value.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: nilf %r2, 5
; CHECK: br %r14
  %and = and i32 %a, 5
  ret i32 %and
}

; Check the highest 16-bit constant that must be handled by NILF.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: nilf %r2, 65533
; CHECK: br %r14
  %and = and i32 %a, 65533
  ret i32 %and
}

; ANDs of 0xffff are zero extensions from i16.
define i32 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %and = and i32 %a, 65535
  ret i32 %and
}

; Check the next value up, which can use RISBG.
define i32 @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: risbg %r2, %r2, 47, 175, 0
; CHECK: br %r14
  %and = and i32 %a, 65536
  ret i32 %and
}

; Check the next value up, which must again use NILF.
define i32 @f9(i32 %a) {
; CHECK-LABEL: f9:
; CHECK: nilf %r2, 65537
; CHECK: br %r14
  %and = and i32 %a, 65537
  ret i32 %and
}

; This value is in range of NILH, but we use RISBG instead.
define i32 @f10(i32 %a) {
; CHECK-LABEL: f10:
; CHECK: risbg %r2, %r2, 47, 191, 0
; CHECK: br %r14
  %and = and i32 %a, 131071
  ret i32 %and
}

; Check the lowest useful NILH value.
define i32 @f11(i32 %a) {
; CHECK-LABEL: f11:
; CHECK: nilh %r2, 2
; CHECK: br %r14
  %and = and i32 %a, 196607
  ret i32 %and
}

; Check the highest useful NILH value.
define i32 @f12(i32 %a) {
; CHECK-LABEL: f12:
; CHECK: nilh %r2, 65530
; CHECK: br %r14
  %and = and i32 %a, -327681
  ret i32 %and
}

; Check the equivalent of NILH of 65531, which can use RISBG.
define i32 @f13(i32 %a) {
; CHECK-LABEL: f13:
; CHECK: risbg %r2, %r2, 46, 172, 0
; CHECK: br %r14
  %and = and i32 %a, -262145
  ret i32 %and
}

; ...same for 65532.
define i32 @f14(i32 %a) {
; CHECK-LABEL: f14:
; CHECK: risbg %r2, %r2, 48, 173, 0
; CHECK: br %r14
  %and = and i32 %a, -196609
  ret i32 %and
}

; ...and 65533.
define i32 @f15(i32 %a) {
; CHECK-LABEL: f15:
; CHECK: risbg %r2, %r2, 47, 173, 0
; CHECK: br %r14
  %and = and i32 %a, -131073
  ret i32 %and
}

; Check the highest useful NILF value.
define i32 @f16(i32 %a) {
; CHECK-LABEL: f16:
; CHECK: nilf %r2, 4294901758
; CHECK: br %r14
  %and = and i32 %a, -65538
  ret i32 %and
}

; Check the next value up, which is the equivalent of an NILH of 65534.
; We use RISBG instead.
define i32 @f17(i32 %a) {
; CHECK-LABEL: f17:
; CHECK: risbg %r2, %r2, 48, 174, 0
; CHECK: br %r14
  %and = and i32 %a, -65537
  ret i32 %and
}

; Check the next value up, which can also use RISBG.
define i32 @f18(i32 %a) {
; CHECK-LABEL: f18:
; CHECK: risbg %r2, %r2, 32, 175, 0
; CHECK: br %r14
  %and = and i32 %a, -65536
  ret i32 %and
}

; ...and again.
define i32 @f19(i32 %a) {
; CHECK-LABEL: f19:
; CHECK: risbg %r2, %r2, 63, 175, 0
; CHECK: br %r14
  %and = and i32 %a, -65535
  ret i32 %and
}

; Check the next value up again, which is the lowest useful NILL value.
define i32 @f20(i32 %a) {
; CHECK-LABEL: f20:
; CHECK: nill %r2, 2
; CHECK: br %r14
  %and = and i32 %a, -65534
  ret i32 %and
}

; Check the highest useful NILL value.
define i32 @f21(i32 %a) {
; CHECK-LABEL: f21:
; CHECK: nill %r2, 65530
; CHECK: br %r14
  %and = and i32 %a, -6
  ret i32 %and
}

; Check the next value up, which can use RISBG.
define i32 @f22(i32 %a) {
; CHECK-LABEL: f22:
; CHECK: risbg %r2, %r2, 62, 188, 0
; CHECK: br %r14
  %and = and i32 %a, -5
  ret i32 %and
}

; ...and again.
define i32 @f23(i32 %a) {
; CHECK-LABEL: f23:
; CHECK: risbg %r2, %r2, 32, 189, 0
; CHECK: br %r14
  %and = and i32 %a, -4
  ret i32 %and
}

; ...and again.
define i32 @f24(i32 %a) {
; CHECK-LABEL: f24:
; CHECK: risbg %r2, %r2, 63, 189, 0
; CHECK: br %r14
  %and = and i32 %a, -3
  ret i32 %and
}

; Check the last useful mask.
define i32 @f25(i32 %a) {
; CHECK-LABEL: f25:
; CHECK: risbg %r2, %r2, 32, 190, 0
; CHECK: br %r14
  %and = and i32 %a, -2
  ret i32 %and
}
