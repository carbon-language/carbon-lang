; RUN: opt %loadPolly -polly-codegen-isl -S -polly-delinearize < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen-isl -S -polly-delinearize -polly-codegen-scev < %s | FileCheck %s

; Verify that we generate the runtime check code after the conditional branch
; in the SCoP region entering block (here %entry).
;
; CHECK: entry:
; CHECK: zext i32 %n to i64
; CHECK: br i1 false
;
; CHECK: %[[T0:[._a-zA-Z0-9]]] = zext i32 %n to i64
; CHECK: %[[T1:[._a-zA-Z0-9]]] = icmp sge i64 %[[T0]], 1
; CHECK: %[[T2:[._a-zA-Z0-9]]] = select i1 %[[T1]], i64 1, i64 0
; CHECK: %[[T3:[._a-zA-Z0-9]]] = icmp ne i64 %[[T2]], 0

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @init_array(i32 %n, double* %data) {
entry:
  %0 = zext i32 %n to i64
  br i1 false, label %for.end10, label %for.body4

for.body4:                                        ; preds = %for.body4, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.body4 ], [ 0, %entry ]
  %.moved.to.for.body4 = mul i64 %0, %indvar1
  %1 = add i64 %.moved.to.for.body4, 0
  %arrayidx7 = getelementptr double* %data, i64 %1
  store double undef, double* %arrayidx7, align 8
  %indvar.next2 = add i64 %indvar1, 1
  br i1 false, label %for.body4, label %for.end10

for.end10:                                        ; preds = %for.body4
  ret void
}
