;; Date: May 27, 2003.
;; From: test/Programs/MultiSource/McCat-05-eks/QRfact.c
;; Function: Matrix QRiterate(Matrix A, Matrix U)
;; 
;; Error: llc produces an invalid register <NULL VALUE> for the
;;        a boolean value computed using setne with a double.
;;
;; Cause: In SparcInstrSelection.cpp, for SetCC, when a result of setne
;;        is used for a branch, it can generate a "branch-on-integer-register"
;;        for integer registers.  In that case, it never saves the value of
;;        the boolean result.  It was attempting to do the same thing for an
;;        FP compare!
;; 
;; LLC Output:
;; !****** Outputing Function: QRiterate_1 ******
;; 
;;         .section ".text"
;;         .align  4
;;         .global QRiterate_1
;;         .type   QRiterate_1, 2
;; QRiterate_1:
;; .L_QRiterate_1_LL_0:
;;         save    %o6, -192, %o6
;;         sethi   %hh(LLVMGlobal__2), %o1
;;         sethi   %lm(LLVMGlobal__2), %o0
;;         or      %o1, %hm(LLVMGlobal__2), %o1
;;         sllx    %o1, 32, %o1
;;         or      %o0, %o1, %o0
;;         or      %o0, %lo(LLVMGlobal__2), %o0
;;         ldd     [%o0+0], %f32
;;         ba      .L_QRiterate_1_LL_1
;;         fcmpd   %fcc0, %f0, %f32
;; 
;; .L_QRiterate_1_LL_1:
;;         brnz    <NULL VALUE>, .L_QRiterate_1_LL_1
;;         nop     
;;         ba      .L_QRiterate_1_LL_2
;;         nop     
;; 
;; .L_QRiterate_1_LL_2:
;;         jmpl    %i7+8, %g0
;;         restore %g0, 0, %g0
;; 
;; .EndOf_QRiterate_1:
;;         .size QRiterate_1, .EndOf_QRiterate_1-QRiterate_1
;;  

target endian = big
target pointersize = 64

implementation   ; Functions:

internal void %QRiterate(double %tmp.212) { 
entry:          ; No predecessors!
        br label %shortcirc_next.1

shortcirc_next.1:               ; preds = %entry
        %tmp.213 = setne double %tmp.212, 0.000000e+00
        br bool %tmp.213, label %shortcirc_next.1, label %exit.1

exit.1:
	ret void
}
