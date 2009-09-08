; RUN: opt < %s -instcombine -S | grep fsub | count 2
; PR4374

define float @func(float %a, float %b) nounwind {
        %tmp3 = fsub float %a, %b
        %tmp4 = fsub float -0.000000e+00, %tmp3
        ret float %tmp4
}
