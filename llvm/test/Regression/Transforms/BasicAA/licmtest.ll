; Test that LICM uses basicaa to do alias analysis, which is capable of 
; disambiguating some obvious cases.  The ToRemove load should be eliminated
; in this testcase.  This testcase was carefully contrived so that GCSE would
; not be able to eliminate the load itself, without licm's help.  This is 
; because, for GCSE, the load is killed by the dummy basic block.

; RUN: if as < %s | opt -basicaa -licm -load-vn -gcse -instcombine | dis | grep ToRemove
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%A = global int 7
%B = global int 8
implementation

int %test(bool %c) {
	%Atmp = load int* %A
	br bool %c, label %Dummy, label %Loop
Loop:
	%ToRemove = load int* %A
	store int %Atmp, int* %B  ; Store cannot alias %A

	br bool %c, label %Out, label %Loop
Out:
	%X = sub int %ToRemove, %Atmp
	ret int %X

Dummy:
	store int 7, int* %A
	br label %Loop
}

