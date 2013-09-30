; RUN: opt < %s  -slp-vectorizer -S -mtriple=i386--netbsd -mcpu=i486
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386--netbsd"

@a = common global i32* null, align 4

; Function Attrs: noreturn nounwind readonly
define i32 @fn1() #0 {
entry:
  %0 = load i32** @a, align 4, !tbaa !4
  %1 = load i32* %0, align 4, !tbaa !5
  %arrayidx1 = getelementptr inbounds i32* %0, i32 1
  %2 = load i32* %arrayidx1, align 4, !tbaa !5
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %c.0 = phi i32 [ %2, %entry ], [ %add2, %do.body ]
  %b.0 = phi i32 [ %1, %entry ], [ %add, %do.body ]
  %add = add nsw i32 %b.0, %c.0
  %add2 = add nsw i32 %add, 1
  br label %do.body
}

attributes #0 = { noreturn nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"int", metadata !1}
!4 = metadata !{metadata !0, metadata !0, i64 0}
!5 = metadata !{metadata !3, metadata !3, i64 0}
