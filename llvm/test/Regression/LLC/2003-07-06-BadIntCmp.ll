;; Date: May 28, 2003.
;; From: test/Programs/MultiSource/Olden-perimeter/maketree.c
;; Function: int CheckOutside(int x, int y)
;; 
;; Note: The .ll code below for this regression test has identical
;;	 behavior to the above function up to the error, but then prints
;; 	 true/false on the two branches.
;; 
;; Error: llc generates a branch-on-xcc instead of branch-on-icc, which
;;        is wrong because the value being compared (int euclid = x*x + y*y)
;;	  overflows, so that the 64-bit and 32-bit compares are not equal.
;; 
;; LLC Output:
;; 
;; !****** Outputing Function: CheckOutside_34 ******
;;         .align  4
;;         .global CheckOutside_34
;;         .type   CheckOutside_34, 2
;; CheckOutside_34:
;; .L_CheckOutside_34_LL_0:
;;         save    %o6, -192, %o6
;;         mulx    %i1, %i1, %i1
;;         mulx    %i0, %i0, %i0
;;         sethi   %lm(4194304), %o1
;;         or      %g0, 1, %o0
;;         add     %i0, %i1, %i0
;;         sra     %o1, 0, %o1
;;         subcc   %i0, %o1, %g0
;;         bg      %xcc, .L_CheckOutside_34_LL_3
;;         nop     
;;         ba      .L_CheckOutside_34_LL_1
;;         nop     
;; ...
;; 
;;


target endian = little
target pointersize = 32
%.str_1 = internal constant [6 x sbyte] c"true\0A\00"		; <[6 x sbyte]*> [#uses=1]
%.str_2 = internal constant [7 x sbyte] c"false\0A\00"		; <[7 x sbyte]*> [#uses=1]

implementation   ; Functions:

declare int %printf(sbyte*, ...)

internal void %__main() {
entry:		; No predecessors!
	ret void
}

internal void %CheckOutside(int %x.1, int %y.1) {
entry:		; No predecessors!
	%tmp.2 = mul int %x.1, %x.1		; <int> [#uses=1]
	%tmp.5 = mul int %y.1, %y.1		; <int> [#uses=1]
	%tmp.6 = add int %tmp.2, %tmp.5		; <int> [#uses=1]
	%tmp.8 = setle int %tmp.6, 4194304		; <bool> [#uses=1]
	br bool %tmp.8, label %then, label %else

then:		; preds = %entry
	%tmp.11 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([6 x sbyte]* %.str_1, long 0, long 0) )		; <int> [#uses=0]
	br label %UnifiedExitNode

else:		; preds = %entry
	%tmp.13 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([7 x sbyte]* %.str_2, long 0, long 0) )		; <int> [#uses=0]
	br label %UnifiedExitNode

UnifiedExitNode:		; preds = %then, %else
	ret void
}

int %main() {
entry:		; No predecessors!
	call void %__main( )
	call void %CheckOutside( int 2097152, int 2097152 )
	ret int 0
}
