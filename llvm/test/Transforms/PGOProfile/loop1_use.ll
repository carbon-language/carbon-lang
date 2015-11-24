; RUN: llvm-profdata merge %S/Inputs/loop1.proftext -o %T/loop1.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/loop1.profdata -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z15test_simple_fori(i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end
; CHECK: !prof !0

for.body:
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
  ret i32 %sum
}

; CHECK: !0 = !{!"branch_weights", i32 96, i32 4}
