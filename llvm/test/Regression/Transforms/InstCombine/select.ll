; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep select

implementation

int %test1(int %A, int %B) {
	%C = select bool false, int %A, int %B
	ret int %C
}

int %test2(int %A, int %B) {
	%C = select bool true, int %A, int %B
	ret int %C
}


