; RUN: opt %loadPolly -polly-parallel \
; RUN: -polly-parallel-force -polly-codegen -S -verify-dom-info < %s \
; RUN: | FileCheck %s -check-prefix=IR

; #define N 1024
; float A[N];
;
; void single_parallel_loop(float alpha) {
;   for (long i = 0; i < N; i++)
;     A[i] = alpha;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; Ensure the scalars are initialized before the OpenMP code is launched.

; IR-LABEL: polly.start:
; IR-NEXT:    store float %alpha, float* %alpha.s2a

; IR: GOMP_parallel_loop_runtime_start

@A = common global [1024 x float] zeroinitializer, align 16

define void @single_parallel_loop(float %alpha) nounwind {
entry:
  br label %for.i

for.i:
  %indvar = phi i64 [ %indvar.next, %for.inc], [ 0, %entry ]
  %scevgep = getelementptr [1024 x float], [1024 x float]* @A, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %S, label %exit

S:
  %alphaplus = fadd float 1.0, %alpha
  store float %alphaplus, float* %scevgep
  br label %for.inc

for.inc:
  %indvar.next = add i64 %indvar, 1
  br label %for.i

exit:
  ret void
}
