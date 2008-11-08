; RUN: llvm-as < %s | llc -march=ppc32 | not grep fneg

; FSEL is not IEEE-safe, and should not be generated except with
; -enable-finite-only-fp-math.  Further, it can't be generated for
; GT ever; it tests for GE.  Leaving in the test for now as it may
; be useful as a basis for a correct test.
;
; XFAIL: *

define double @test_FNEG_sel(double %A, double %B, double %C) {
        %D = sub double -0.000000e+00, %A               ; <double> [#uses=1]
        %Cond = fcmp ugt double %D, -0.000000e+00               ; <i1> [#uses=1]
        %E = select i1 %Cond, double %B, double %C              ; <double> [#uses=1]
        ret double %E
}

