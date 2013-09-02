; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-codegen < %s

; This test case checks that Polly does not break for PHI guard statement.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @phi_guard() {
entry:
  %acc.reg2mem = alloca i32
  br label %for.preheader

for.preheader:                              ; preds = %for.end, %entry
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.end ]
  store i32 0, i32* %acc.reg2mem
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %for.preheader
  %0 = phi i32 [ 0, %for.preheader ], [ %1, %for.inc ]
  %1 = add nsw i32 %0, 1
  store i32 %1, i32* %acc.reg2mem
  %exitcond = icmp ne i32 %1, 20
  br i1 %exitcond, label %for.inc, label %for.end

for.end:                                          ; preds = %for.inc
  %indvar.next = add i64 %indvar, 1
  %exitcond4 = icmp ne i64 %indvar.next, 20
  br i1 %exitcond4, label %for.preheader, label %for.end10

for.end10:                                        ; preds = %for.end
  %res = load i32* %acc.reg2mem, align 4
  ret i32 %res
}
