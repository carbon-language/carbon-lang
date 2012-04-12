; RUN: opt %loadPolly -basicaa -polly-codegen -enable-polly-vector -dce -S %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x float] zeroinitializer, align 16
@B = common global [1024 x float**] zeroinitializer, align 16

declare float** @foo(float) readnone

define void @simple_vec_call() nounwind {
entry:
  br label %body

body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar_next, %body ]
  %scevgep = getelementptr [1024 x float**]* @B, i64 0, i64 %indvar
  %value = load float* getelementptr inbounds ([1024 x float]* @A, i64 0, i64 0), align 16
  %result = tail call float** @foo(float %value) nounwind
  store float** %result, float*** %scevgep, align 4
  %indvar_next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar_next, 4
  br i1 %exitcond, label %return, label %body

return:
  ret void
}

; CHECK: %p_scevgep = getelementptr [1024 x float**]* @B, i64 0, i64 0
; CHECK: %p_scevgep1 = getelementptr [1024 x float**]* @B, i64 0, i64 1
; CHECK: %p_scevgep2 = getelementptr [1024 x float**]* @B, i64 0, i64 2
; CHECK: %p_scevgep3 = getelementptr [1024 x float**]* @B, i64 0, i64 3
; CHECK: %value_p_splat_one = load <1 x float>* bitcast ([1024 x float]* @A to <1 x float>*), align 8
; CHECK: %value_p_splat = shufflevector <1 x float> %value_p_splat_one, <1 x float> %value_p_splat_one, <4 x i32> zeroinitializer
; CHECK: %0 = extractelement <4 x float> %value_p_splat, i32 0
; CHECK: %1 = extractelement <4 x float> %value_p_splat, i32 1
; CHECK: %2 = extractelement <4 x float> %value_p_splat, i32 2
; CHECK: %3 = extractelement <4 x float> %value_p_splat, i32 3
; CHECK: %p_result = tail call float** @foo(float %0) nounwind
; CHECK: %p_result4 = tail call float** @foo(float %1) nounwind
; CHECK: %p_result5 = tail call float** @foo(float %2) nounwind
; CHECK: %p_result6 = tail call float** @foo(float %3) nounwind
; CHECK: store float** %p_result, float*** %p_scevgep, align 4
; CHECK: store float** %p_result4, float*** %p_scevgep1, align 4
; CHECK: store float** %p_result5, float*** %p_scevgep2, align 4
; CHECK: store float** %p_result6, float*** %p_scevgep3, align 4
