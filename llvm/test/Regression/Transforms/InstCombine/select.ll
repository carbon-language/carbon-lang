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

int %test13(int %a, int %b) {
	%C = seteq int %a, %b
	%V = select bool %C, int %a, int %b
	ret int %V
}

int %test13a(int %a, int %b) {
	%C = setne int %a, %b
	%V = select bool %C, int %a, int %b
	ret int %V
}

int %test13b(int %a, int %b) {
	%C = seteq int %a, %b
	%V = select bool %C, int %b, int %a
	ret int %V
}

bool %test14a(bool %C, int %X) {
	%V = select bool %C, int %X, int 0
	%R = setlt int %V, 1                  ; (X < 1) | !C
	ret bool %R
}

bool %test14b(bool %C, int %X) {
	%V = select bool %C, int 0, int %X
	%R = setlt int %V, 1                  ; (X < 1) | C
	ret bool %R
}

int %test15a(int %X) {       ;; Code sequence for (X & 16) ? 16 : 0
        %t1 = and int %X, 16
        %t2 = seteq int %t1, 0
        %t3 = select bool %t2, int 0, int 16 ;; X & 16
        ret int %t3
}

int %test15b(int %X) {       ;; Code sequence for (X & 32) ? 0 : 24
        %t1 = and int %X, 32
        %t2 = seteq int %t1, 0
        %t3 = select bool %t2, int 32, int 0 ;; ~X & 32
        ret int %t3
}

int %test15c(int %X) {       ;; Alternate code sequence for (X & 16) ? 16 : 0
        %t1 = and int %X, 16
        %t2 = seteq int %t1, 16
        %t3 = select bool %t2, int 16, int 0 ;; X & 16
        ret int %t3
}

int %test15d(int %X) {       ;; Alternate code sequence for (X & 16) ? 16 : 0
        %t1 = and int %X, 16
        %t2 = setne int %t1, 0
        %t3 = select bool %t2, int 16, int 0 ;; X & 16
        ret int %t3
}
