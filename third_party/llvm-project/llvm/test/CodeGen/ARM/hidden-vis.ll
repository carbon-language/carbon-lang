; RUN: llc < %s -mtriple=arm-linux | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s -check-prefix=DARWIN

@a = hidden global i32 0
@b = external global i32

define weak hidden void @t1() nounwind {
; LINUX: .hidden t1
; LINUX-LABEL: t1:

; DARWIN: .private_extern _t1
; DARWIN-LABEL: t1:
  ret void
}

define weak void @t2() nounwind {
; LINUX-LABEL: t2:
; LINUX: .hidden a

; DARWIN-LABEL: t2:
; DARWIN: .private_extern _a
  ret void
}
