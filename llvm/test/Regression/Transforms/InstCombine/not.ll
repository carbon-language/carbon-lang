; This test makes sure that these instructions are properly eliminated.
;

; RUN: as < %s | opt -instcombine -die | dis | not grep xor

implementation

int %test1(int %A) {
	%B = xor int %A, -1
	%C = xor int %B, -1
	ret int %C
}

bool %test2(int %A, int %B) {
	%cond = setle int %A, %B     ; Can change into setge
	%Ret = xor bool %cond, true
	ret bool %Ret
}


; Test that demorgans law can be instcombined
int %test3(int %A, int %B) {
	%a = xor int %A, -1
	%b = xor int %B, -1
	%c = and int %a, %b
	%d = xor int %c, -1
	ret int %d
}

; Test that demorgens law can work with constants
int %test4(int %A, int %B) {
	%a = xor int %A, -1
	%c = and int %a, 5    ; 5 = ~c2
	%d = xor int %c, -1
	ret int %d
}

; test the mirror of demorgans law...
int %test5(int %A, int %B) {
	%a = xor int %A, -1
	%b = xor int %B, -1
	%c = or int %a, %b
	%d = xor int %c, -1
	ret int %d
}
