; Test that LICM uses basicaa to do alias analysis, which is capable of 
; disambiguating some obvious cases.  If LICM is able to disambiguate the
; two pointers, then the load should be hoisted, and the store sunk.  Thus
; the loop becomes empty and can be deleted by ADCE. 

; RUN: llvm-as < %s | opt -basicaa -licm --adce | llvm-dis | not grep Loop

%A = global int 7
%B = global int 8
%C = global [2 x int ] [ int 4, int 8 ]
implementation

int %test(bool %c) {
	%Atmp = load int* %A
	br label %Loop
Loop:
	%ToRemove = load int* %A
	store int %Atmp, int* %B  ; Store cannot alias %A

	br bool %c, label %Out, label %Loop
Out:
	%X = sub int %ToRemove, %Atmp
	ret int %X
}

int %test2(bool %c) {
	br label %Loop
Loop:
	%AVal = load int* %A
	%C0 = getelementptr [2 x int ]* %C, long 0, long 0
	store int %AVal, int* %C0  ; Store cannot alias %A

	%BVal = load int* %B
	%C1 = getelementptr [2 x int ]* %C, long 0, long 1
	store int %BVal, int* %C1  ; Store cannot alias %A, %B, or %C0

	br bool %c, label %Out, label %Loop
Out:
	%X = sub int %AVal, %BVal
	ret int %X
}

