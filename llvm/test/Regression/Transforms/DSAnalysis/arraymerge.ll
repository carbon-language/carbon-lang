; This testcase is used to make sure that the outer element of arrays are 
; folded completely away if possible.  This is a very common case, so it should
; be efficient.
;
; RUN: analyze %s -tddatastructure
;
implementation

sbyte* %merge1([100 x sbyte] *%A, long %N) {
	%P = getelementptr [100 x sbyte] *%A, long 0, long %N
	ret sbyte* %P
}

sbyte* %merge2([100 x sbyte] *%A, long %N) {
        ; The graph for this example should end up exactly the same as for merge1
        %P1 = getelementptr [100 x sbyte] *%A, long 0, long 0
	%P2 = getelementptr sbyte* %P1, long %N
	ret sbyte* %P2
}

