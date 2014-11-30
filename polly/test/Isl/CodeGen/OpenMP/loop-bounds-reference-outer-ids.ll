; RUN: opt %loadPolly -polly-parallel -polly-ast -analyze < %s | FileCheck %s -check-prefix=AST
; RUN: opt %loadPolly -polly-parallel -polly-codegen-isl -S < %s | FileCheck %s -check-prefix=IR
; RUN: opt %loadPolly -polly-parallel -polly-codegen-isl -S < %s | FileCheck %s -check-prefix=IR
;
; float A[100];
;
; void loop_references_outer_ids(long n) {
;   for (long i = 0; i < 100; i++)
;     for (long j = 0; j < 100; j++)
;       for (long k = 0; k < n + i; k++)
;         A[j] += i + j + k;
; }

; In this test case we verify that the j-loop is generated as OpenMP parallel
; loop and that the values of 'i' and 'n', needed in the loop bounds of the
; k-loop, are correctly passed to the subfunction.

; AST: #pragma minimal dependence distance: 1
; AST: for (int c1 = max(0, -n + 1); c1 <= 99; c1 += 1)
; AST:   #pragma omp parallel for
; AST:   for (int c3 = 0; c3 <= 99; c3 += 1)
; AST:     #pragma minimal dependence distance: 1
; AST:     for (int c5 = 0; c5 < n + c1; c5 += 1)
; AST:       Stmt_for_body6(c1, c3, c5);

; IR: %polly.par.userContext = alloca { i64, i64 }
; IR: %4 = bitcast { i64, i64 }* %polly.par.userContext to i8*
; IR-NEXT: call void @llvm.lifetime.start(i64 16, i8* %4)
; IR-NEXT: %5 = getelementptr inbounds { i64, i64 }* %polly.par.userContext, i32 0, i32 0
; IR-NEXT: store i64 %n, i64* %5
; IR-NEXT: %6 = getelementptr inbounds { i64, i64 }* %polly.par.userContext, i32 0, i32 1
; IR-NEXT: store i64 %polly.indvar, i64* %6
; IR-NEXT: %polly.par.userContext1 = bitcast { i64, i64 }* %polly.par.userContext to i8*

; IR-LABEL: @loop_references_outer_ids.polly.subfn(i8* %polly.par.userContext)
; IR: %polly.par.userContext1 = bitcast i8* %polly.par.userContext to { i64, i64 }*
; IR-NEXT: %0 = getelementptr inbounds { i64, i64 }* %polly.par.userContext1, i32 0, i32 0
; IR-NEXT: %1 = load i64* %0
; IR-NEXT: %2 = getelementptr inbounds { i64, i64 }* %polly.par.userContext1, i32 0, i32 1
; IR-NEXT: %3 = load i64* %2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [100 x float] zeroinitializer, align 16

define void @loop_references_outer_ids(i64 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc13, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc14, %for.inc13 ]
  %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.end15

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc10, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc11, %for.inc10 ]
  %exitcond = icmp ne i64 %j.0, 100
  br i1 %exitcond, label %for.body3, label %for.end12

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i64 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %add = add nsw i64 %i.0, %n
  %cmp5 = icmp slt i64 %k.0, %add
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %add7 = add nsw i64 %i.0, %j.0
  %add8 = add nsw i64 %add7, %k.0
  %conv = sitofp i64 %add8 to float
  %arrayidx = getelementptr inbounds [100 x float]* @A, i64 0, i64 %j.0
  %tmp = load float* %arrayidx, align 4
  %add9 = fadd float %tmp, %conv
  store float %add9, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nsw i64 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %inc11 = add nsw i64 %j.0, 1
  br label %for.cond1

for.end12:                                        ; preds = %for.cond1
  br label %for.inc13

for.inc13:                                        ; preds = %for.end12
  %inc14 = add nsw i64 %i.0, 1
  br label %for.cond

for.end15:                                        ; preds = %for.cond
  ret void
}
