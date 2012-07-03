; Make sure this testcase codegens to the fabs instruction, not a call to fabsf
; RUN: llc < %s -march=x86 -mattr=-sse2,-sse3,-sse | FileCheck %s
; RUN: llc < %s -march=x86 -mattr=-sse,-sse2,-sse3 -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck %s --check-prefix=UNSAFE

declare float @fabsf(float)

declare x86_fp80 @fabsl(x86_fp80)

; CHECK:  test1:
; UNSAFE: test1:
define float @test1(float %X) {
        %Y = call float @fabsf(float %X)
        ret float %Y
}
; CHECK:  {{^[ \t]+fabs$}}
; UNSAFE: {{^[ \t]+fabs$}}

; CHECK-NOT:  fabs
; UNSAFE-NOT: fabs

; CHECK:  test2:
; UNSAFE: test2:
define double @test2(double %X) {
        %Y = fcmp oge double %X, -0.0
        %Z = fsub double -0.0, %X
        %Q = select i1 %Y, double %X, double %Z
        ret double %Q
}
; fabs is not used here.
; CHECK-NOT:  fabs

; UNSAFE: {{^[ \t]+fabs$}}

; UNSAFE-NOT: fabs

; CHECK:  test3:
; UNSAFE: test3:
define x86_fp80 @test3(x86_fp80 %X) {
        %Y = call x86_fp80 @fabsl(x86_fp80 %X)
        ret x86_fp80 %Y
}
; CHECK:  {{^[ \t]+fabs$}}
; UNSAFE: {{^[ \t]+fabs$}}

; CHECK-NOT:  fabs
; UNSAFE-NOT: fabs
