; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=swift       | FileCheck %s -check-prefix=SWIFT
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 | FileCheck %s -check-prefix=T2
; rdar://12348580

define void @t1() noreturn forcesizeopt nounwind ssp {
entry:
; ARM: t1:
; ARM: bl _bar

; SWIFT: t1:
; SWIFT: bl _bar

; T2: t1:
; T2: blx _bar
  tail call void @bar() noreturn nounwind
  unreachable
}

define void @t2() noreturn forcesizeopt nounwind ssp {
entry:
; ARM: t2:
; ARM: bl _t1

; SWIFT: t2:
; SWIFT: bl _t1

; T2: t2:
; T2: bl _t1
  tail call void @t1() noreturn nounwind
  unreachable
}

declare void @bar() noreturn
