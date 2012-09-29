; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=swift       | FileCheck %s -check-prefix=SWIFT
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 | FileCheck %s -check-prefix=T2
; rdar://8979299

define void @t1() noreturn nounwind ssp {
entry:
; ARM: t1:
; ARM: mov lr, pc
; ARM: b _bar

; SWIFT: t1:
; SWIFT: mov lr, pc
; SWIFT: b _bar

; T2: t1:
; T2: blx _bar
  tail call void @bar() noreturn nounwind
  unreachable
}

define void @t2() noreturn nounwind ssp {
entry:
; ARM: t2:
; ARM: mov lr, pc
; ARM: b _t1

; SWIFT: t2:
; SWIFT: mov lr, pc
; SWIFT: b _t1

; T2: t2:
; T2: mov lr, pc
; T2: b.w _t1
  tail call void @t1() noreturn nounwind
  unreachable
}

declare void @bar() noreturn
