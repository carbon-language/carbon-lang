; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @oppositesign
; CHECK: add nsw i16 %a, %b
define i16 @oppositesign(i16 %x, i16 %y) {
; %a is negative, %b is positive
  %a = or i16 %x, 32768
  %b = and i16 %y, 32767
  %c = add i16 %a, %b
  ret i16 %c
}

; CHECK-LABEL: @ripple_nsw1
; CHECK: add nsw i16 %a, %b
define i16 @ripple_nsw1(i16 %x, i16 %y) {
; %a has at most one bit set
  %a = and i16 %y, 1

; %b has a 0 bit other than the sign bit
  %b = and i16 %x, 49151

  %c = add i16 %a, %b
  ret i16 %c
}

; Like the previous test, but flip %a and %b
; CHECK-LABEL: @ripple_nsw2
; CHECK: add nsw i16 %b, %a
define i16 @ripple_nsw2(i16 %x, i16 %y) {
  %a = and i16 %y, 1
  %b = and i16 %x, 49151
  %c = add i16 %b, %a
  ret i16 %c
}

; CHECK-LABEL: @ripple_no_nsw1
; CHECK: add i32 %a, %x
define i32 @ripple_no_nsw1(i32 %x, i32 %y) {
; We know nothing about %x
  %a = and i32 %y, 1
  %b = add i32 %a, %x
  ret i32 %b
}

; CHECK-LABEL: @ripple_no_nsw2
; CHECK: add i16 %a, %b
define i16 @ripple_no_nsw2(i16 %x, i16 %y) {
; %a has at most one bit set
  %a = and i16 %y, 1

; %b has a 0 bit, but it is the sign bit
  %b = and i16 %x, 32767

  %c = add i16 %a, %b
  ret i16 %c
}
