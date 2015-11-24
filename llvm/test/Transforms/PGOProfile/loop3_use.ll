; RUN: llvm-profdata merge %S/Inputs/loop3.proftext -o %T/loop3.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/loop3.profdata -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z15test_nested_foriii(i32 %r, i32 %s, i32 %t) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  %nested_for_sum.0 = phi i32 [ 1, %entry ], [ %nested_for_sum.1, %for.inc11 ]
  %cmp = icmp slt i32 %i.0, %r
  br i1 %cmp, label %for.body, label %for.end13
; CHECK: !prof !0

for.body:
  br label %for.cond1

for.cond1:
  %j.0 = phi i32 [ 0, %for.body ], [ %inc9, %for.inc8 ]
  %nested_for_sum.1 = phi i32 [ %nested_for_sum.0, %for.body ], [ %nested_for_sum.2, %for.inc8 ]
  %cmp2 = icmp slt i32 %j.0, %s
  br i1 %cmp2, label %for.body3, label %for.end10
; CHECK: !prof !1

for.body3:
  br label %for.cond4

for.cond4:
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc7, %for.inc ]
  %nested_for_sum.2 = phi i32 [ %nested_for_sum.1, %for.body3 ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i32 %k.0, %t
  br i1 %cmp5, label %for.body6, label %for.end
; CHECK: !prof !2

for.body6:
  %inc = add nsw i32 %nested_for_sum.2, 1
  br label %for.inc

for.inc:
  %inc7 = add nsw i32 %k.0, 1
  br label %for.cond4

for.end:
  br label %for.inc8

for.inc8:
  %inc9 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end10:
  br label %for.inc11

for.inc11:
  %inc12 = add nsw i32 %i.0, 1
  br label %for.cond

for.end13:
  ret i32 %nested_for_sum.0
}

; CHECK: !0 = !{!"branch_weights", i32 10, i32 6}
; CHECK: !1 = !{!"branch_weights", i32 33, i32 10}
; CHECK: !2 = !{!"branch_weights", i32 186, i32 33}
