;; Date:     Jul 8, 2003.
;; From:     test/Programs/MultiSource/Olden-perimeter
;; Function: int %adj(uint %d.1, uint %ct.1)
;;
;; Errors: (1) cast-int-to-bool was being treated as a NOP (i.e., the int
;;	       register was treated as effectively true if non-zero).
;;	       This cannot be used for later boolean operations.
;;	   (2) (A or NOT(B)) was being folded into A orn B, which is ok
;;	       for bitwise operations but not booleans!  For booleans,
;;	       the result has to be compared with 0.
;; 
;; LLC Output for the basic block (LLVM assembly is shown below):
;; 
;; .L_adj_7_LL_4:
;;         sethi   0, %i0
;;         subcc   %i1, 2, %g0
;;         move    %icc, 1, %i0
;;         orn     %i0, %i1, %i0
;;         ba      .L_adj_7_LL_5
;;         nop     
;; 


target endian = big
target pointersize = 64

%.str_1 = internal constant [30 x sbyte] c"d = %d, ct = %d, d ^ ct = %d\0A\00"


implementation   ; Functions:

declare int %printf(sbyte*, ...)

int %adj(uint %d.1, uint %ct.1) {
entry:
        %tmp.19 = seteq uint %ct.1, 2
        %tmp.22.not = cast uint %ct.1 to bool
        %tmp.221 = xor bool %tmp.22.not, true
        %tmp.26 = or bool %tmp.19, %tmp.221
        %tmp.27 = cast bool %tmp.26 to int
        ret int %tmp.27
}

int %main() {
entry:
	%result = call int %adj(uint 3, uint 2)
	%tmp.0 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([30 x sbyte]* %.str_1, long 0, long 0), uint 3, uint 2, int %result)
	ret int 0
}
