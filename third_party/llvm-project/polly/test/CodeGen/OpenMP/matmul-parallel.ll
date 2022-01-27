; RUN: opt %loadPolly -polly-parallel -polly-opt-isl -polly-ast -disable-output -debug-only=polly-ast < %s 2>&1 | FileCheck --check-prefix=AST %s
; RUN: opt %loadPolly -polly-parallel -polly-opt-isl -polly-codegen -S < %s | FileCheck --check-prefix=CODEGEN %s
; REQUIRES: asserts

; Parellization of detected matrix-multiplication. The allocations
; Packed_A and Packed_B must be passed to the outlined function.
; llvm.org/PR43164
;
; #define N 1536
; int foo(float A[N][N],float B[N][N],float C[N][N]) {
;     for (int i = 0; i < N; i++) {
;         for (int j = 0; j < N; j++) {
;             for (int k = 0; k < N; k++)
;                 C[i][j] = C[i][j] + A[i][k] * B[k][j];
;         }
;     }
;     return 0;
; }

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.27034"

define i32 @foo([1536 x float]* nocapture readonly %A, [1536 x float]* nocapture readonly %B, [1536 x float]* nocapture %C) {
entry:
  br label %entry.split

entry.split:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv50 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next51, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup:
  ret i32 0

for.cond5.preheader:
  %indvars.iv47 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next48, %for.cond.cleanup7 ]
  %arrayidx10 = getelementptr inbounds [1536 x float], [1536 x float]* %C, i64 %indvars.iv50, i64 %indvars.iv47
  br label %for.body8

for.cond.cleanup3:
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %exitcond52 = icmp eq i64 %indvars.iv.next51, 1536
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:
  %indvars.iv.next48 = add nuw nsw i64 %indvars.iv47, 1
  %exitcond49 = icmp eq i64 %indvars.iv.next48, 1536
  br i1 %exitcond49, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %0 = load float, float* %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [1536 x float], [1536 x float]* %A, i64 %indvars.iv50, i64 %indvars.iv
  %1 = load float, float* %arrayidx14, align 4
  %arrayidx18 = getelementptr inbounds [1536 x float], [1536 x float]* %B, i64 %indvars.iv, i64 %indvars.iv47
  %2 = load float, float* %arrayidx18, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  store float %add, float* %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1536
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8
}


; AST: #pragma omp parallel for

; CODGEN-LABEL: define internal void @init_array_polly_subfn(i8* %polly.par.userContext)
; CODEGEN: %polly.subfunc.arg.Packed_A = load
; CODEGEN: %polly.subfunc.arg.Packed_B = load
