; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info 2>&1 | FileCheck %s

; ModuleID = 'b.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK: NoAlias:   i32* %p, i32* @xyz

;@xyz = global i32 12, align 4
@xyz = internal unnamed_addr global notaddrtaken i32 12, align 4

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i32* nocapture %p, i32* nocapture %q) #0 {
entry:
  %0 = load i32* @xyz, align 4, !tbaa !0
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @xyz, align 4, !tbaa !0
  store i32 1, i32* %p, align 4, !tbaa !0
  %1 = load i32* @xyz, align 4, !tbaa !0
  store i32 %1, i32* %q, align 4, !tbaa !0
  ret i32 undef
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !1, metadata !1, i64 0}
!1 = metadata !{metadata !"int", metadata !2, i64 0}
!2 = metadata !{metadata !"omnipotent char", metadata !3, i64 0}
!3 = metadata !{metadata !"Simple C/C++ TBAA"}
