; Test loads of symbolic addresses when generating small-model non-PIC.
; All addresses can be treated as PC
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@e4 = external global i32
@d4 = global i32 1
@e2 = external global i32, align 2
@d2 = global i32 1, align 2
@e1 = external global i32, align 1
@d1 = global i32 1, align 1

declare void @ef()
define void @df() {
  ret void
}

; Test a load of a fully-aligned external variable.
define i32 *@f1() {
; CHECK: f1:
; CHECK: larl %r2, e4
; CHECK-NEXT: br %r14
  ret i32 *@e4
}

; Test a load of a fully-aligned local variable.
define i32 *@f2() {
; CHECK: f2:
; CHECK: larl %r2, d4
; CHECK-NEXT: br %r14
  ret i32 *@d4
}

; Test a load of a 2-byte-aligned external variable.
define i32 *@f3() {
; CHECK: f3:
; CHECK: larl %r2, e2
; CHECK-NEXT: br %r14
  ret i32 *@e2
}

; Test a load of a 2-byte-aligned local variable.
define i32 *@f4() {
; CHECK: f4:
; CHECK: larl %r2, d2
; CHECK-NEXT: br %r14
  ret i32 *@d2
}

; Test a load of an unaligned external variable, which must go via the GOT.
define i32 *@f5() {
; CHECK: f5:
; CHECK: lgrl %r2, e1@GOT
; CHECK-NEXT: br %r14
  ret i32 *@e1
}

; Test a load of an unaligned local variable, which must go via the GOT.
define i32 *@f6() {
; CHECK: f6:
; CHECK: lgrl %r2, d1@GOT
; CHECK-NEXT: br %r14
  ret i32 *@d1
}

; Test a load of an external function.
define void() *@f7() {
; CHECK: f7:
; CHECK: larl %r2, ef
; CHECK-NEXT: br %r14
  ret void() *@ef
}

; Test a load of a local function.
define void() *@f8() {
; CHECK: f8:
; CHECK: larl %r2, df
; CHECK-NEXT: br %r14
  ret void() *@df
}
