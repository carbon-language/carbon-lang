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

bool %test7(bool %C, bool %X) {
        %R = select bool %C, bool true, bool %X    ; R = or C, X
        ret bool %R
}

bool %test8(bool %C, bool %X) {
        %R = select bool %C, bool %X, bool false   ; R = and C, X
        ret bool %R
}

bool %test9(bool %C, bool %X) {
        %R = select bool %C, bool false, bool %X    ; R = and !C, X
        ret bool %R
}

bool %test10(bool %C, bool %X) {
        %R = select bool %C, bool %X, bool true   ; R = or !C, X
        ret bool %R
}

int %test11(int %a) {
        %C = seteq int %a, 0
        %R = select bool %C, int 0, int 1
        ret int %R
}

int %test12(bool %cond, int %a) {
	%b = or int %a, 1
	%c = select bool %cond, int %b, int %a
	ret int %c
}

int %test12a(bool %cond, int %a) {
	%b = shr int %a, ubyte 1
	%c = select bool %cond, int %b, int %a
	ret int %c
}

int %test12b(bool %cond, int %a) {
	%b = shr int %a, ubyte 1
	%c = select bool %cond, int %a, int %b
	ret int %c
}

