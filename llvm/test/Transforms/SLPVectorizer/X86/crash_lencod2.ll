; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Function Attrs: nounwind ssp uwtable
define void @intrapred_luma() #0 {
entry:
  %conv153 = trunc i32 undef to i16
  %arrayidx154 = getelementptr inbounds [13 x i16]* undef, i64 0, i64 12
  store i16 %conv153, i16* %arrayidx154, align 8, !tbaa !0
  %arrayidx155 = getelementptr inbounds [13 x i16]* undef, i64 0, i64 11
  store i16 %conv153, i16* %arrayidx155, align 2, !tbaa !0
  %arrayidx156 = getelementptr inbounds [13 x i16]* undef, i64 0, i64 10
  store i16 %conv153, i16* %arrayidx156, align 4, !tbaa !0
  ret void
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !"short", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
