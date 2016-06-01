; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Type = type { [10 x i32], i32 }
@Foo = external global %struct.Type, align 4

; /// Check that BasicAA claims no alias between different fileds of a structure
; void test() {
;   for (unsigned i = 0 ; i < 10 ; i++) 
;     Foo.arr[i] += Foo.i;
; }

define void @test() {
; CHECK-LABEL: Function: test:
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.Type, %struct.Type* @Foo, i64 0, i32 1), align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]

  %arrayidx = getelementptr inbounds %struct.Type, %struct.Type* @Foo, i64 0, i32 0, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %arrayidx, align 4
; CHECK: NoAlias: i32* %arrayidx, i32* getelementptr inbounds (%struct.Type, %struct.Type* @Foo, i64 0, i32 1)

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
