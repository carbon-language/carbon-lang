; PR1075
; RUN: llc < %s -mtriple=x86_64-apple-darwin -O3 | FileCheck %s

define float @foo(float %x) nounwind {
    %tmp1 = fmul float %x, 3.000000e+00
    %tmp3 = fmul float %x, 5.000000e+00
    %tmp5 = fmul float %x, 7.000000e+00
    %tmp7 = fmul float %x, 1.100000e+01
    %tmp10 = fadd float %tmp1, %tmp3
    %tmp12 = fadd float %tmp10, %tmp5
    %tmp14 = fadd float %tmp12, %tmp7
    ret float %tmp14

; CHECK: mulss	LCPI0_0(%rip)
; CHECK: mulss	LCPI0_1(%rip)
; CHECK: addss
; CHECK: mulss	LCPI0_2(%rip)
; CHECK: addss
; CHECK: mulss	LCPI0_3(%rip)
; CHECK: addss
; CHECK: ret
}
