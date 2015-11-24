; RUN: llvm-profdata merge %S/Inputs/branch2.proftext -o %T/branch2.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/branch2.profdata -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z9test_br_2i(i32 %i) {
entry:
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else
; CHECK: !prof !0

if.then:
  %add = add nsw i32 %i, 2
  br label %if.end

if.else:
  %sub = sub nsw i32 %i, 2
  br label %if.end

if.end:
  %retv = phi i32 [ %add, %if.then ], [ %sub, %if.else ]
  ret i32 %retv
}

; CHECK: !0 = !{!"branch_weights", i32 1, i32 1}
