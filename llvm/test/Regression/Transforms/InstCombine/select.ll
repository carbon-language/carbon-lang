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

int %test3(bool %C, int %I) {
	%V = select bool %C, int %I, int %I         ; V = I
	ret int %V
}

bool %test4(bool %C) {
	%V = select bool %C, bool true, bool false  ; V = C
	ret bool %V
}

bool %test5(bool %C) {
	%V = select bool %C, bool false, bool true  ; V = !C
	ret bool %V
}

int %test6(bool %C) {
	%V = select bool %C, int 1, int 0         ; V = cast C to int
	ret int %V
}

