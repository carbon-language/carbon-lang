;; Date:     May 27, 2003.
;; From:     Variant of 2003-05-27-usefsubasbool.ll
;; 
;; Error: llc fails to save a boolean value in a register (and later uses an
;;	  invalid register <NULL VALUE> in a BRNZ) for a boolean value
;;        used only by branches but in a different basic block.
;;
;; Cause: In SparcInstrSelection.cpp, for SetCC, when a result of setCC
;;        is used only for branches, it is not saved into an int. register.
;;        But if the boolean is used in a branch in a different basic block,
;;        that branch uses a BRNZ inst. instead of a branch-on-CC.
;; 
;; LLC Output before fix:
;; !****** Outputing Function: QRiterate_1 ******
;; 
;;         .section ".text"
;;         .align  4
;;         .global QRiterate_1
;;         .type   QRiterate_1, 2
;; QRiterate_1:
;; .L_QRiterate_1_LL_0:
;;         save    %o6, -192, %o6
;;         sethi   %lm(LLVMGlobal__2), %o2
;;         sethi   %hh(LLVMGlobal__2), %o1
;;         or      %o1, %hm(LLVMGlobal__2), %o1
;;         sllx    %o1, 32, %o1
;;         or      %o2, %o1, %o2
;;         or      %o2, %lo(LLVMGlobal__2), %o2
;;         ldd     [%o2+0], %f32
;;         fcmpd   %fcc0, %f0, %f32
;;         ba      .L_QRiterate_1_LL_1
;;         nop     
;; 
;; .L_QRiterate_1_LL_1:
;;         brnz    <NULL_VALUE>, .L_QRiterate_1_LL_1
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
        %tmp.213 = setne double %tmp.212, 0.000000e+00
        br label %shortcirc_next.1

shortcirc_next.1:               ; preds = %entry
        br bool %tmp.213, label %shortcirc_next.1, label %exit.1

exit.1:
	ret void
}
