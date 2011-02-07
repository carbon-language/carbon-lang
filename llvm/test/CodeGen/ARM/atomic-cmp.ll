; RUN: llc < %s -mtriple=armv7-apple-darwin   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s -check-prefix=T2
; rdar://8964854

define i8 @t(i8* %a, i8 %b, i8 %c) nounwind {
; ARM: t:
; ARM: ldrexb
; ARM: strexb

; T2: t:
; T2: ldrexb
; T2: strexb
  %tmp0 = tail call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* %a, i8 %b, i8 %c)
  ret i8 %tmp0
}

declare i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* nocapture, i8, i8) nounwind
