; RUN: opt -enable-tbaa -basicaa -tbaa -gvn -instcombine -S < %s | grep {ret i32 0}
; RUN: opt -enable-tbaa -tbaa -basicaa -gvn -instcombine -S < %s | grep {ret i32 1075000115}

; According to the TBAA metadata the load and store don't alias. However,
; according to the actual code, they do. The order of the alias analysis
; passes should determine which of these takes precedence.

target datalayout = "e-p:64:64:64"

define i32 @trouble(i32* %x) nounwind ssp {
entry:
  store i32 0, i32* %x, !tbaa !0
  %0 = bitcast i32* %x to float*
  store float 0x4002666660000000, float* %0, !tbaa !3
  %tmp3 = load i32* %x, !tbaa !0
  ret i32 %tmp3
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"simple"}
!3 = metadata !{metadata !"float", metadata !1}
