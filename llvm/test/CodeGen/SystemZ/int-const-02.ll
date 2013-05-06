; Test loading of 64-bit constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check 0.
define i64 @f1() {
; CHECK: f1:
; CHECK: lghi %r2, 0
; CHECK-NEXT: br %r14
  ret i64 0
}

; Check the high end of the LGHI range.
define i64 @f2() {
; CHECK: f2:
; CHECK: lghi %r2, 32767
; CHECK-NEXT: br %r14
  ret i64 32767
}

; Check the next value up, which must use LLILL instead.
define i64 @f3() {
; CHECK: f3:
; CHECK: llill %r2, 32768
; CHECK-NEXT: br %r14
  ret i64 32768
}

; Check the high end of the LLILL range.
define i64 @f4() {
; CHECK: f4:
; CHECK: llill %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 65535
}

; Check the first useful LLILH value, which is the next one up.
define i64 @f5() {
; CHECK: f5:
; CHECK: llilh %r2, 1
; CHECK-NEXT: br %r14
  ret i64 65536
}

; Check the first useful LGFI value, which is the next one up again.
define i64 @f6() {
; CHECK: f6:
; CHECK: lgfi %r2, 65537
; CHECK-NEXT: br %r14
  ret i64 65537
}

; Check the high end of the LGFI range.
define i64 @f7() {
; CHECK: f7:
; CHECK: lgfi %r2, 2147483647
; CHECK-NEXT: br %r14
  ret i64 2147483647
}

; Check the next value up, which should use LLILH instead.
define i64 @f8() {
; CHECK: f8:
; CHECK: llilh %r2, 32768
; CHECK-NEXT: br %r14
  ret i64 2147483648
}

; Check the next value up again, which should use LLILF.
define i64 @f9() {
; CHECK: f9:
; CHECK: llilf %r2, 2147483649
; CHECK-NEXT: br %r14
  ret i64 2147483649
}

; Check the high end of the LLILH range.
define i64 @f10() {
; CHECK: f10:
; CHECK: llilh %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 4294901760
}

; Check the next value up, which must use LLILF.
define i64 @f11() {
; CHECK: f11:
; CHECK: llilf %r2, 4294901761
; CHECK-NEXT: br %r14
  ret i64 4294901761
}

; Check the high end of the LLILF range.
define i64 @f12() {
; CHECK: f12:
; CHECK: llilf %r2, 4294967295
; CHECK-NEXT: br %r14
  ret i64 4294967295
}

; Check the lowest useful LLIHL value, which is the next one up.
define i64 @f13() {
; CHECK: f13:
; CHECK: llihl %r2, 1
; CHECK-NEXT: br %r14
  ret i64 4294967296
}

; Check the next value up, which must use a combination of two instructions.
define i64 @f14() {
; CHECK: f14:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oill %r2, 1
; CHECK-NEXT: br %r14
  ret i64 4294967297
}

; Check the high end of the OILL range.
define i64 @f15() {
; CHECK: f15:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oill %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 4295032831
}

; Check the next value up, which should use OILH instead.
define i64 @f16() {
; CHECK: f16:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilh %r2, 1
; CHECK-NEXT: br %r14
  ret i64 4295032832
}

; Check the next value up again, which should use OILF.
define i64 @f17() {
; CHECK: f17:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilf %r2, 65537
; CHECK-NEXT: br %r14
  ret i64 4295032833
}

; Check the high end of the OILH range.
define i64 @f18() {
; CHECK: f18:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilh %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 8589869056
}

; Check the high end of the OILF range.
define i64 @f19() {
; CHECK: f19:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilf %r2, 4294967295
; CHECK-NEXT: br %r14
  ret i64 8589934591
}

; Check the high end of the LLIHL range.
define i64 @f20() {
; CHECK: f20:
; CHECK: llihl %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 281470681743360
}

; Check the lowest useful LLIHH value, which is 1<<32 greater than the above.
define i64 @f21() {
; CHECK: f21:
; CHECK: llihh %r2, 1
; CHECK-NEXT: br %r14
  ret i64 281474976710656
}

; Check the lowest useful LLIHF value, which is 1<<32 greater again.
define i64 @f22() {
; CHECK: f22:
; CHECK: llihf %r2, 65537
; CHECK-NEXT: br %r14
  ret i64 281479271677952
}

; Check the highest end of the LLIHH range.
define i64 @f23() {
; CHECK: f23:
; CHECK: llihh %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 -281474976710656
}

; Check the next value up, which must use OILL too.
define i64 @f24() {
; CHECK: f24:
; CHECK: llihh %r2, 65535
; CHECK-NEXT: oill %r2, 1
; CHECK-NEXT: br %r14
  ret i64 -281474976710655
}

; Check the high end of the LLIHF range.
define i64 @f25() {
; CHECK: f25:
; CHECK: llihf %r2, 4294967295
; CHECK-NEXT: br %r14
  ret i64 -4294967296
}

; Check -1.
define i64 @f26() {
; CHECK: f26:
; CHECK: lghi %r2, -1
; CHECK-NEXT: br %r14
  ret i64 -1
}

; Check the low end of the LGHI range.
define i64 @f27() {
; CHECK: f27:
; CHECK: lghi %r2, -32768
; CHECK-NEXT: br %r14
  ret i64 -32768
}

; Check the next value down, which must use LGFI instead.
define i64 @f28() {
; CHECK: f28:
; CHECK: lgfi %r2, -32769
; CHECK-NEXT: br %r14
  ret i64 -32769
}

; Check the low end of the LGFI range.
define i64 @f29() {
; CHECK: f29:
; CHECK: lgfi %r2, -2147483648
; CHECK-NEXT: br %r14
  ret i64 -2147483648
}

; Check the next value down, which needs a two-instruction sequence.
define i64 @f30() {
; CHECK: f30:
; CHECK: llihf %r2, 4294967295
; CHECK-NEXT: oilf %r2, 2147483647
; CHECK-NEXT: br %r14
  ret i64 -2147483649
}
