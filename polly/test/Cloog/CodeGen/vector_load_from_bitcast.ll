; RUN: opt %loadPolly -tbaa -polly-codegen -polly-vectorizer=polly -S < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.8.0"

define void @foo(float** %tone)  {
preheader:
  br label %for

for:
  %indvar = phi i32 [ %indvar.next, %for ], [ 0, %preheader ]
  %ptr1 = getelementptr inbounds float** %tone, i32 0
  %val = load float** %ptr1, align 4, !tbaa !2817
  %ptr2 = getelementptr float* %val, i32 %indvar
  %re115 = bitcast float* %ptr2 to float*
; CHECK: %val_p_vec_p = bitcast float** %p_ptr1 to <1 x float*>*

  store float undef, float* %re115, !tbaa !2816
; CHECK: store <4 x float> undef, <4 x float>* %vector_ptr

  %indvar.next = add nsw i32 %indvar, 1
  %exitcond = icmp ne i32 %indvar.next, 4
  br i1 %exitcond, label %for, label %end

end:
  unreachable
}

!2814 = metadata !{metadata !"Simple C/C++ TBAA"}
!2813 = metadata !{metadata !"omnipotent char", metadata !2814, i64 0}
!2816 = metadata !{metadata !"int", metadata !2813, i64 0}
!2817 = metadata !{metadata !"short", metadata !2813, i64 0}
