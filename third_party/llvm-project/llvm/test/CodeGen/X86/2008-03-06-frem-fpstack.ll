; RUN: llc < %s -mtriple=i686-- -mcpu=i386
; PR2122
define float @func(float %a, float %b) nounwind  {
entry:
        %tmp3 = frem float %a, %b               ; <float> [#uses=1]
        ret float %tmp3
}
