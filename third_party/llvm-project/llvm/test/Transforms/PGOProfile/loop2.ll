; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry=false -S | FileCheck %s --check-prefixes=GEN,NOTENTRY
; RUN: llvm-profdata merge %S/Inputs/loop2.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=false -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-gen -pgo-instrument-entry=true -S | FileCheck %s --check-prefixes=GEN,ENTRY
; RUN: llvm-profdata merge %S/Inputs/loop2_entry.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_test_nested_for = private constant [15 x i8] c"test_nested_for"

define i32 @test_nested_for(i32 %r, i32 %s) {
entry:
; GEN: entry:
; NOTENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_nested_for, i64 {{[0-9]+}}, i32 3, i32 2)
; ENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_nested_for, i64 {{[0-9]+}}, i32 3, i32 0)
  br label %for.cond.outer

for.cond.outer:
; GEN: for.cond.outer:
; GEN-NOT: call void @llvm.instrprof.increment
  %i.0 = phi i32 [ 0, %entry ], [ %inc.2, %for.inc.outer ]
  %sum.0 = phi i32 [ 1, %entry ], [ %sum.1, %for.inc.outer ]
  %cmp = icmp slt i32 %i.0, %r
  br i1 %cmp, label %for.body.outer, label %for.end.outer
; USE: br i1 %cmp, label %for.body.outer, label %for.end.outer
; USE-SAME: !prof ![[BW_FOR_COND_OUTER:[0-9]+]]

for.body.outer:
; GEN: for.body.outer:
; GEN-NOT: call void @llvm.instrprof.increment
  br label %for.cond.inner

for.cond.inner:
; GEN: for.cond.inner:
; GEN-NOT: call void @llvm.instrprof.increment
  %j.0 = phi i32 [ 0, %for.body.outer ], [ %inc.1, %for.inc.inner ]
  %sum.1 = phi i32 [ %sum.0, %for.body.outer ], [ %inc, %for.inc.inner ]
  %cmp2 = icmp slt i32 %j.0, %s
  br i1 %cmp2, label %for.body.inner, label %for.end.inner
; USE: br i1 %cmp2, label %for.body.inner, label %for.end.inner
; USE-SAME: !prof ![[BW_FOR_COND_INNER:[0-9]+]]

for.body.inner:
; GEN: for.body.inner:
; GEN-NOT: call void @llvm.instrprof.increment
  %inc = add nsw i32 %sum.1, 1
  br label %for.inc.inner

for.inc.inner:
; GEN: for.inc.inner:
; NOTENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_nested_for, i64 {{[0-9]+}}, i32 3, i32 0)
; ENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_nested_for, i64 {{[0-9]+}}, i32 3, i32 1)
  %inc.1 = add nsw i32 %j.0, 1
  br label %for.cond.inner

for.end.inner:
; GEN: for.end.inner:
  br label %for.inc.outer

for.inc.outer:
; GEN: for.inc.outer:
; NOTENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_nested_for, i64 {{[0-9]+}}, i32 3, i32 1)
; ENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_nested_for, i64 {{[0-9]+}}, i32 3, i32 2)
  %inc.2 = add nsw i32 %i.0, 1
  br label %for.cond.outer

for.end.outer:
; GEN: for.end.outer:
; GEN-NOT: call void @llvm.instrprof.increment
; GEN: ret i32
  ret i32 %sum.0
}

; USE-DAG: ![[BW_FOR_COND_OUTER]] = !{!"branch_weights", i32 10, i32 6}
; USE-DAG: ![[BW_FOR_COND_INNER]] = !{!"branch_weights", i32 33, i32 10}

