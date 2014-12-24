; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm "pselect = __pselect"
; CHECK: pselect = __pselect

; CHECK: pselect:
; CHECK: retq
define void @pselect() {
  ret void
}
