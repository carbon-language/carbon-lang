; Test that GCSE uses basicaa to do alias analysis, which is capable of 
; disambiguating some obvious cases.  All loads should be removable in 
; this testcase.

; RUN: if as < %s | opt -basicaa -load-vn -gcse -instcombine -dce | dis | grep load
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%A = global int 7
%B = global int 8
implementation

int %test() {
	%A1 = load int* %A

	store int 123, int* %B  ; Store cannot alias %A

	%A2 = load int* %A
	%X = sub int %A1, %A2
	ret int %X
}

int %test2() {
        %A1 = load int* %A
        br label %Loop
Loop:
        %AP = phi int [0, %0], [%X, %Loop]
        store int %AP, int* %B  ; Store cannot alias %A

        %A2 = load int* %A
        %X = sub int %A1, %A2
        %c = seteq int %X, 0
        br bool %c, label %out, label %Loop

out:
        ret int %X
}

