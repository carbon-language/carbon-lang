; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register (r11 for PPC).

define i8* @nest_receiver(i8* nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK: # %bb.0:
; CHECK-NEXT: mr 3, 11
; CHECK-NEXT: blr

  ret i8* %arg
}

define i8* @nest_caller(i8* %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mr 11, 3
; CHECK-NEXT: bl nest_receiver
; CHECK: blr

  %result = call i8* @nest_receiver(i8* nest %arg)
  ret i8* %result
}

define void @test_indirect(i32 ()* nocapture %f, i8* %p) {
entry:

; CHECK-LABEL: test_indirect
; CHECK-DAG: ld [[DEST:[0-9]+]], 0(3)
; CHECK-DAG: ld 2, 8(3)
; CHECK-DAG: mr 11, 4
; CHECK: mtctr [[DEST]]
; CHECK: bctrl
; CHECK: blr

  %callee.knr.cast = bitcast i32 ()* %f to i32 (i8*)*
  %call = tail call signext i32 %callee.knr.cast(i8* nest %p)
  ret void
}

