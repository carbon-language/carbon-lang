; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1738
define i1 @test1(double %X, double %Y) {
        %tmp9 = fcmp ord double %X, 0.000000e+00
        %tmp13 = fcmp ord double %Y, 0.000000e+00
        %bothcond = and i1 %tmp13, %tmp9
        ret i1 %bothcond
; CHECK:  fcmp ord double %Y, %X
}
