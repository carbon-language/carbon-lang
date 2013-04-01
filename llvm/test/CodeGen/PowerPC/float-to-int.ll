; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i64 @foo(float %a) nounwind {
  %x = fptosi float %a to i64
  ret i64 %x

; CHECK: @foo
; CHECK: fctidz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr
}

define i64 @foo2(double %a) nounwind {
  %x = fptosi double %a to i64
  ret i64 %x

; CHECK: @foo2
; CHECK: fctidz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr
}

define i64 @foo3(float %a) nounwind {
  %x = fptoui float %a to i64
  ret i64 %x

; CHECK: @foo3
; CHECK: fctiduz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr
}

define i64 @foo4(double %a) nounwind {
  %x = fptoui double %a to i64
  ret i64 %x

; CHECK: @foo4
; CHECK: fctiduz [[REG:[0-9]+]], 1
; CHECK: stfd [[REG]],
; CHECK: ld 3,
; CHECK: blr
}

define i32 @goo(float %a) nounwind {
  %x = fptosi float %a to i32
  ret i32 %x

; CHECK: @goo
; CHECK: fctiwz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr
}

define i32 @goo2(double %a) nounwind {
  %x = fptosi double %a to i32
  ret i32 %x

; CHECK: @goo2
; CHECK: fctiwz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr
}

define i32 @goo3(float %a) nounwind {
  %x = fptoui float %a to i32
  ret i32 %x

; CHECK: @goo3
; CHECK: fctiwuz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr
}

define i32 @goo4(double %a) nounwind {
  %x = fptoui double %a to i32
  ret i32 %x

; CHECK: @goo4
; CHECK: fctiwuz [[REG:[0-9]+]], 1
; CHECK: stfiwx [[REG]],
; CHECK: lwz 3,
; CHECK: blr
}

