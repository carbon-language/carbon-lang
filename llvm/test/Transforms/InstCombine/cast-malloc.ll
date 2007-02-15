; test that casted mallocs get converted to malloc of the right type
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    not grep bitcast

; The target datalayout is important for this test case. We have to tell 
; instcombine that the ABI alignment for a long is 4-bytes, not 8, otherwise
; it won't do the transform.
target datalayout = "e-i64:32:64"
int* %test(uint %size) {
	%X = malloc long, uint %size
        %ret = bitcast long* %X to int*
	ret int* %ret
}
