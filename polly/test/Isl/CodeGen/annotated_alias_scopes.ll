; RUN: opt %loadPolly -polly-code-generator=isl -polly-codegen-isl -S < %s | FileCheck %s --check-prefix=SCOPES
; RUN: opt %loadPolly -polly-code-generator=isl -polly-codegen-isl -polly-annotate-alias-scopes=false -S < %s | FileCheck %s --check-prefix=NOSCOPES
;
; Check that we create alias scopes that indicate the accesses to A, B and C cannot alias in any way.
;
; SCOPES:      %[[BIdx:[._a-zA-Z0-9]*]] = getelementptr inbounds i32* %B, i64 %polly.indvar
; SCOPES:      load i32* %[[BIdx]], align 4, !alias.scope ![[AliasScopeB:[0-9]*]], !noalias ![[NoAliasB:[0-9]*]]
; SCOPES:      %[[CIdx:[._a-zA-Z0-9]*]] = getelementptr inbounds float* %C, i64 %polly.indvar
; SCOPES:      load float* %[[CIdx]], align 4, !alias.scope ![[AliasScopeC:[0-9]*]], !noalias ![[NoAliasC:[0-9]*]]
; SCOPES:      %[[AIdx:[._a-zA-Z0-9]*]] = getelementptr inbounds i32* %A, i64 %polly.indvar
; SCOPES:      store i32 %{{[._a-zA-Z0-9]*}}, i32* %[[AIdx]], align 4, !alias.scope ![[AliasScopeA:[0-9]*]], !noalias ![[NoAliasA:[0-9]*]]
;
; SCOPES:      ![[AliasScopeB]] = metadata !{metadata ![[AliasScopeB]], metadata !{{[0-9]*}}, metadata !"polly.alias.scope.B"}
; SCOPES:      ![[NoAliasB]] = metadata !{
; SCOPES-DAG:     metadata ![[AliasScopeA]]
; SCOPES-DAG:     metadata ![[AliasScopeC]]
; SCOPES:       }
; SCOPES-DAG:  ![[AliasScopeA]] = metadata !{metadata ![[AliasScopeA]], metadata !{{[0-9]*}}, metadata !"polly.alias.scope.A"}
; SCOPES-DAG:  ![[AliasScopeC]] = metadata !{metadata ![[AliasScopeC]], metadata !{{[0-9]*}}, metadata !"polly.alias.scope.C"}
; SCOPES:      ![[NoAliasC]] = metadata !{
; SCOPES-DAG:     metadata ![[AliasScopeA]]
; SCOPES-DAG:     metadata ![[AliasScopeB]]
; SCOPES:       }
; SCOPES:      ![[NoAliasA]] = metadata !{
; SCOPES-DAG:     metadata ![[AliasScopeB]]
; SCOPES-DAG:     metadata ![[AliasScopeC]]
; SCOPES:       }
;
; NOSCOPES:    %[[BIdx:[._a-zA-Z0-9]*]] = getelementptr inbounds i32* %B, i64 %polly.indvar
; NOSCOPES:    load i32* %[[BIdx]]
; NOSCOPES-NOT: alias.scope
; NOSCOPES-NOT: noalias
; NOSCOPES:    %[[CIdx:[._a-zA-Z0-9]*]] = getelementptr inbounds float* %C, i64 %polly.indvar
; NOSCOPES:    load float* %[[CIdx]]
; NOSCOPES-NOT: alias.scope
; NOSCOPES-NOT: noalias
; NOSCOPES:    %[[AIdx:[._a-zA-Z0-9]*]] = getelementptr inbounds i32* %A, i64 %polly.indvar
; NOSCOPES:    store i32 %{{[._a-zA-Z0-9]*}}, i32* %[[AIdx]]
; NOSCOPES-NOT: alias.scope
; NOSCOPES-NOT: noalias
;
; NOSCOPES-NOT: metadata
;
;    void jd(int *A, int *B, float *C) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[i] + C[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32* %B, float* %C) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv
  %tmp = load i32* %arrayidx, align 4
  %conv = sitofp i32 %tmp to float
  %arrayidx2 = getelementptr inbounds float* %C, i64 %indvars.iv
  %tmp1 = load float* %arrayidx2, align 4
  %add = fadd fast float %conv, %tmp1
  %conv3 = fptosi float %add to i32
  %arrayidx5 = getelementptr inbounds i32* %A, i64 %indvars.iv
  store i32 %conv3, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
