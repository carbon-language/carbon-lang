; This testcase tests for various features the basicaa test should be able to 
; determine, as noted in the comments.

; RUN: if as < %s | opt -basicaa -load-vn -gcse -instcombine -dce | dis | grep REMOVE
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi


; Array test:  Test that operations on one local array do not invalidate 
; operations on another array.  Important for scientific codes.
;
int %different_array_test(long %A, long %B) {
	%Array1 = alloca int, uint 100
	%Array2 = alloca int, uint 200

	%pointer = getelementptr int* %Array1, long %A
	%val = load int* %pointer

	%pointer2 = getelementptr int* %Array2, long %B
	store int 7, int* %pointer2

	%REMOVE = load int* %pointer ; redundant with above load
	%retval = sub int %REMOVE, %val
	ret int %retval
}

; Constant index test: Constant indexes into the same array should not 
; interfere with each other.  Again, important for scientific codes.
;
int %constant_array_index_test() {
	%Array = alloca int, uint 100
	%P1 = getelementptr int* %Array, long 7
	%P2 = getelementptr int* %Array, long 6
	
	%A = load int* %P1
	store int 1, int* %P2   ; Should not invalidate load
	%BREMOVE = load int* %P1
	%Val = sub int %A, %BREMOVE
	ret int %Val
}

; Test that if two pointers are spaced out by a constant getelementptr, that 
; they cannot alias.
int %gep_distance_test(int* %A) {
        %REMOVEu = load int* %A
        %B = getelementptr int* %A, long 2  ; Cannot alias A
        store int 7, int* %B
        %REMOVEv = load int* %A
        %r = sub int %REMOVEu, %REMOVEv
        ret int %r
}

; Test that if two pointers are spaced out by a constant offset, that they
; cannot alias, even if there is a variable offset between them...
int %gep_distance_test2({int,int}* %A, long %distance) {
	%A = getelementptr {int,int}* %A, long 0, ubyte 0
	%REMOVEu = load int* %A
	%B = getelementptr {int,int}* %A, long %distance, ubyte 1
	store int 7, int* %B    ; B cannot alias A, it's at least 4 bytes away
	%REMOVEv = load int* %A
        %r = sub int %REMOVEu, %REMOVEv
        ret int %r
}

int %foo(int * %A) {
	%X = load int* %A
	%B = cast int* %A to sbyte*
	%C = getelementptr sbyte* %B, long 4
	%Y = load sbyte* %C
	ret int 8
}
