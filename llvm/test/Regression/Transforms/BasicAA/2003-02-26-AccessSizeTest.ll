; This testcase makes sure that size is taken to account when alias analysis 
; is performed.  It is not legal to delete the second load instruction because
; the value computed by the first load instruction is changed by the store.

; RUN: llvm-as < %s | opt -load-vn -gcse -instcombine | llvm-dis | grep DONOTREMOVE

int %test() {
	%A = alloca int
	store int 0, int* %A
        %X = load int* %A
        %B = cast int* %A to sbyte*
        %C = getelementptr sbyte* %B, long 1
	store sbyte 1, sbyte* %C    ; Aliases %A
        %Y.DONOTREMOVE = load int* %A
	%Z = sub int %X, %Y.DONOTREMOVE
        ret int %Z
}

