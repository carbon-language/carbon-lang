;; Date: May 28, 2003.
;; From: test/Programs/MultiSource/McCat-05-eks/QRfact.c
;; Function: Matrix QRiterate(Matrix A, Matrix U)
;; 
;; Error: llc produces an invalid register <NULL VALUE> for the
;;        phi argument %tmp.213 produced by fcmpd:
;; 
;; LLC Output:
;; 
;; !****** Outputing Function: QRiterate_1 ******
;; 
;;         .section ".text"
;;         .align  4
;;         .global QRiterate_1
;;         .type   QRiterate_1, 2
;; QRiterate_1:
;; .L_QRiterate_1_LL_0:
;;         save    %o6, -192, %o6
;;         brgz    %i0, .L_QRiterate_1_LL_1
;;         add     %g0, %g0, %o0
;;         ba      .L_QRiterate_1_LL_2
;;         nop     
;; 
;; .L_QRiterate_1_LL_1:
;;         sethi   %lm(LLVMGlobal__2), %o1
;;         sethi   %hh(LLVMGlobal__2), %o0
;;         or      %o0, %hm(LLVMGlobal__2), %o0
;;         sllx    %o0, 32, %o0
;;         or      %o1, %o0, %o1
;;         or      %o1, %lo(LLVMGlobal__2), %o1
;;         ldd     [%o1+0], %f32
;;         fcmpd   %fcc0, %f2, %f32
;;         ba      .L_QRiterate_1_LL_2
;;         add     <NULL VALUE>, %g0, %o0
;; 
;; .L_QRiterate_1_LL_2:
;;         brnz    %o0, .L_QRiterate_1_LL_1
;;         nop     
;;         ba      .L_QRiterate_1_LL_3
;;         nop     
;;
;; .L_QRiterate_1_LL_3:
;;         jmpl    %i7+8, %g0
;;         restore %g0, 0, %g0
;; 
;; .EndOf_QRiterate_1:
;;         .size QRiterate_1, .EndOf_QRiterate_1-QRiterate_1
;; 


target endian = big
target pointersize = 64

implementation   ; Functions:

internal void %QRiterate(int %p.1, double %tmp.212) { 
entry:          ; No predecessors!
        %tmp.184 = setgt int %p.1, 0            ; <bool> [#uses=1]
        br bool %tmp.184, label %shortcirc_next.1, label %shortcirc_done.1

shortcirc_next.1:               ; preds = %entry
        %tmp.213 = setne double %tmp.212, 0.000000e+00
        br label %shortcirc_done.1

shortcirc_done.1:               ; preds = %entry, %shortcirc_next.1
        %val.1 = phi bool [ false, %entry ], [ %tmp.213, %shortcirc_next.1 ]
        br bool %val.1, label %shortcirc_next.1, label %exit.1

exit.1:
	ret void
}
