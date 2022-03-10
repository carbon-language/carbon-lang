; RUN: llc < %s -relocation-model=pic -mattr=+sse2 | FileCheck %s

; We used to do tail calls through the GOT for these symbols, but it was
; disabled due to PR15086.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-unknown-freebsd9.0"

define double @test1(double %x) nounwind readnone {
; CHECK-LABEL: test1:
; CHECK:  calll foo@PLT
  %1 = tail call double @foo(double %x) nounwind readnone
  ret double %1
}

declare double @foo(double) readnone

define double @test2(double %x) nounwind readnone {
; CHECK-LABEL: test2:
; CHECK:  calll sin@PLT
  %1 = tail call double @sin(double %x) nounwind readnone
  ret double %1
}

declare double @sin(double) readnone

define double @test3(double %x) nounwind readnone {
; CHECK-LABEL: test3:
; CHECK:  calll sin2@PLT
  %1 = tail call double @sin2(double %x) nounwind readnone
  ret double %1
}

declare double @sin2(double) readnone
