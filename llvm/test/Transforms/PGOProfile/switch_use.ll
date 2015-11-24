; RUN: llvm-profdata merge %S/Inputs/switch.proftext -o %T/switch.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/switch.profdata -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z13test_switch_1i(i32 %i) {
entry:
  switch i32 %i, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
  ]
; CHECK: !prof !0

sw.bb:
  %add = add nsw i32 %i, 2
  br label %sw.epilog

sw.bb1:
  %add2 = add nsw i32 %i, 100
  br label %sw.epilog

sw.bb3:
  %add4 = add nsw i32 %i, 4
  br label %sw.epilog

sw.default:
  %add5 = add nsw i32 %i, 1
  br label %sw.epilog

sw.epilog:
  %retv = phi i32 [ %add5, %sw.default ], [ %add4, %sw.bb3 ], [ %add2, %sw.bb1 ], [ %add, %sw.bb ]
  ret i32 %retv
}

;CHECK: !0 = !{!"branch_weights", i32 3, i32 0, i32 0, i32 0}
