; PR1075
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | \
; RUN:   %prcontext {mulss	LCPI1_3} 1 | grep mulss | count 1

define float @foo(float %x) {
    %tmp1 = fmul float %x, 3.000000e+00
    %tmp3 = fmul float %x, 5.000000e+00
    %tmp5 = fmul float %x, 7.000000e+00
    %tmp7 = fmul float %x, 1.100000e+01
    %tmp10 = fadd float %tmp1, %tmp3
    %tmp12 = fadd float %tmp10, %tmp5
    %tmp14 = fadd float %tmp12, %tmp7
    ret float %tmp14
}
