; RUN: not llc -no-integrated-as %s -o - 2> %t1
; RUN: FileCheck %s < %t1
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--"

; CHECK: error: couldn't allocate output register for constraint '{ax}'
define i128 @blup() {
  %v = tail call i128 asm "", "={ax},0,~{dirflag},~{fpsr},~{flags}"(i128 0)
  ret i128 %v
}
