; Test loads of symbolic addresses when generating medium- and
; large-model non-PIC.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -code-model=medium | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -code-model=large | FileCheck %s

@ev = external global i32
@dv = global i32 0
@pv = protected global i32 0
@hv = hidden global i32 0

declare void @ef()
define void @df() {
  ret void
}
define protected void @pf() {
  ret void
}
define hidden void @hf() {
  ret void
}

; Test loads of external variables.  There is no guarantee that the
; variable will be in range of LARL.
define i32 *@f1() {
; CHECK-LABEL: f1:
; CHECK: lgrl %r2, ev@GOT
; CHECK: br %r14
  ret i32 *@ev
}

; ...likewise locally-defined normal-visibility variables.
define i32 *@f2() {
; CHECK-LABEL: f2:
; CHECK: lgrl %r2, dv@GOT
; CHECK: br %r14
  ret i32 *@dv
}

; ...likewise protected variables.
define i32 *@f3() {
; CHECK-LABEL: f3:
; CHECK: lgrl %r2, pv@GOT
; CHECK: br %r14
  ret i32 *@pv
}

; ...likewise hidden variables.
define i32 *@f4() {
; CHECK-LABEL: f4:
; CHECK: lgrl %r2, hv@GOT
; CHECK: br %r14
  ret i32 *@hv
}

; Check loads of external functions.  This could use LARL, but we don't have
; code to detect that yet.
define void() *@f5() {
; CHECK-LABEL: f5:
; CHECK: lgrl %r2, ef@GOT
; CHECK: br %r14
  ret void() *@ef
}

; ...likewise locally-defined normal-visibility functions.
define void() *@f6() {
; CHECK-LABEL: f6:
; CHECK: lgrl %r2, df@GOT
; CHECK: br %r14
  ret void() *@df
}

; ...likewise protected functions.
define void() *@f7() {
; CHECK-LABEL: f7:
; CHECK: lgrl %r2, pf@GOT
; CHECK: br %r14
  ret void() *@pf
}

; ...likewise hidden functions.
define void() *@f8() {
; CHECK-LABEL: f8:
; CHECK: lgrl %r2, hf@GOT
; CHECK: br %r14
  ret void() *@hf
}
