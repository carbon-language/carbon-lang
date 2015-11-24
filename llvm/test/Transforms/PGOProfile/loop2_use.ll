; RUN: llvm-profdata merge %S/Inputs/loop2.proftext -o %T/loop2.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/loop2.profdata -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z13test_do_whilei(i32 %n) {
entry:
  br label %do.body

do.body:
  %i.0 = phi i32 [ 0, %entry ], [ %inc1, %do.cond ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %do.cond ]
  %inc = add nsw i32 %sum, 1
  br label %do.cond

do.cond:
  %inc1 = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %do.body, label %do.end
; CHECK: !prof !0

do.end:
  ret i32 %inc
}

; CHECK: !0 = !{!"branch_weights", i32 92, i32 4}
