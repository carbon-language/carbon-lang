; RUN: llvm-profdata merge %S/Inputs/single_bb.proftext -o %T/single_bb.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/single_bb.profdata -S 2>&1 | FileCheck %s

; CHECK: Function control flow change detected (hash mismatch) _Z9single_bbv
; CHECK: No profile data available for function _ZL8uncalledii

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z9single_bbv() {
entry:
  ret i32 0
}

define i32 @_ZL8uncalledii(i32 %i, i32 %j) {
  %mul = mul nsw i32 %i, %j
  ret i32 %mul
}
