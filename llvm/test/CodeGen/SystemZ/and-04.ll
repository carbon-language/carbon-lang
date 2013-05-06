; Test 64-bit ANDs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; There is no 64-bit AND instruction for a mask of 1.
; FIXME: we ought to be able to require "ngr %r2, %r0", but at the moment,
; two-address optimisations force "ngr %r0, %r2; lgr %r2, %r0" instead.
define i64 @f1(i64 %a) {
; CHECK: f1:
; CHECK: lghi %r0, 1
; CHECK: ngr
; CHECK: br %r14
  %and = and i64 %a, 1
  ret i64 %and
}

; Likewise 0xfffe.
define i64 @f2(i64 %a) {
; CHECK: f2:
; CHECK: llill %r0, 65534
; CHECK: ngr
; CHECK: br %r14
  %and = and i64 %a, 65534
  ret i64 %and
}

; ...but 0xffff is a 16-bit zero extension.
define i64 @f3(i64 %a) {
; CHECK: f3:
; CHECK: llghr %r2, %r2
; CHECK: br %r14
  %and = and i64 %a, 65535
  ret i64 %and
}

; Check the next value up, which again has no dedicated instruction.
define i64 @f4(i64 %a) {
; CHECK: f4:
; CHECK: llilh %r0, 1
; CHECK: ngr
; CHECK: br %r14
  %and = and i64 %a, 65536
  ret i64 %and
}

; Check 0xfffffffe.
define i64 @f5(i64 %a) {
; CHECK: f5:
; CHECK: lilf %r0, 4294967294
; CHECK: ngr
; CHECK: br %r14
  %and = and i64 %a, 4294967294
  ret i64 %and
}

; Check the next value up, which is a 32-bit zero extension.
define i64 @f6(i64 %a) {
; CHECK: f6:
; CHECK: llgfr %r2, %r2
; CHECK: br %r14
  %and = and i64 %a, 4294967295
  ret i64 %and
}

; Check the lowest useful NIHF value (0x00000001_ffffffff).
define i64 @f7(i64 %a) {
; CHECK: f7:
; CHECK: nihf %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 8589934591
  ret i64 %and
}

; Check the low end of the NIHH range (0x0000ffff_ffffffff).
define i64 @f8(i64 %a) {
; CHECK: f8:
; CHECK: nihh %r2, 0
; CHECK: br %r14
  %and = and i64 %a, 281474976710655
  ret i64 %and
}

; Check the highest useful NIHH value (0xfffeffff_ffffffff).
define i64 @f9(i64 %a) {
; CHECK: f9:
; CHECK: nihh %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, -281474976710657
  ret i64 %and
}

; Check the highest useful NIHF value (0xfffefffe_ffffffff).
define i64 @f10(i64 %a) {
; CHECK: f10:
; CHECK: nihf %r2, 4294901758
; CHECK: br %r14
  %and = and i64 %a, -281479271677953
  ret i64 %and
}

; Check the low end of the NIHL range (0xffff0000_ffffffff).
define i64 @f11(i64 %a) {
; CHECK: f11:
; CHECK: nihl %r2, 0
; CHECK: br %r14
  %and = and i64 %a, -281470681743361
  ret i64 %and
}

; Check the highest useful NIHL value (0xfffffffe_ffffffff).
define i64 @f12(i64 %a) {
; CHECK: f12:
; CHECK: nihl %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, -4294967297
  ret i64 %and
}

; Check the low end of the NILF range (0xffffffff_00000000).
define i64 @f13(i64 %a) {
; CHECK: f13:
; CHECK: nilf %r2, 0
; CHECK: br %r14
  %and = and i64 %a, -4294967296
  ret i64 %and
}

; Check the low end of the NILH range (0xffffffff_0000ffff).
define i64 @f14(i64 %a) {
; CHECK: f14:
; CHECK: nilh %r2, 0
; CHECK: br %r14
  %and = and i64 %a, -4294901761
  ret i64 %and
}

; Check the next value up, which must use NILF.
define i64 @f15(i64 %a) {
; CHECK: f15:
; CHECK: nilf %r2, 65536
; CHECK: br %r14
  %and = and i64 %a, -4294901760
  ret i64 %and
}

; Check the maximum useful NILF value (0xffffffff_fffefffe).
define i64 @f16(i64 %a) {
; CHECK: f16:
; CHECK: nilf %r2, 4294901758
; CHECK: br %r14
  %and = and i64 %a, -65538
  ret i64 %and
}

; Check the highest useful NILH value, which is one greater than the above.
define i64 @f17(i64 %a) {
; CHECK: f17:
; CHECK: nilh %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, -65537
  ret i64 %and
}

; Check the low end of the NILL range, which is one greater again.
define i64 @f18(i64 %a) {
; CHECK: f18:
; CHECK: nill %r2, 0
; CHECK: br %r14
  %and = and i64 %a, -65536
  ret i64 %and
}

; Check the highest useful NILL value.
define i64 @f19(i64 %a) {
; CHECK: f19:
; CHECK: nill %r2, 65534
; CHECK: br %r14
  %and = and i64 %a, -2
  ret i64 %and
}
