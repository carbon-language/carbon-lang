; This test makes sure that these instructions are properly eliminated.
;

; RUN: as < %s | opt -instcombine | dis | not grep set

%X = uninitialized global int

bool %test1(int %A) {
	%B = seteq int %A, %A
	%C = seteq int* %X, null   ; Never true
	%D = and bool %B, %C
	ret bool %D
}

bool %test2(int %A) {
	%B = setne int %A, %A
	%C = setne int* %X, null   ; Never false
	%D = or bool %B, %C
	ret bool %D
}

bool %test3(int %A) {
	%B = setlt int %A, %A
	ret bool %B
}

bool %test4(int %A) {
	%B = setgt int %A, %A
	ret bool %B
}

bool %test5(int %A) {
	%B = setle int %A, %A
	ret bool %B
}

bool %test6(int %A) {
	%B = setge int %A, %A
	ret bool %B
}

bool %test7(uint %A) {
	%B = setge uint %A, 0  ; true
	ret bool %B
}

bool %test8(uint %A) {
	%B = setlt uint %A, 0  ; false
	ret bool %B
}

;; test operations on boolean values these should all be eliminated$a
bool %test9(bool %A) {
	%B = setlt bool %A, false ; false
	ret bool %B
}
bool %test10(bool %A) {
	%B = setgt bool %A, true  ; false
	ret bool %B
}
bool %test11(bool %A) {
	%B = setle bool %A, true ; true
	ret bool %B
}
bool %test12(bool %A) {
	%B = setge bool %A, false  ; true
	ret bool %B
}
bool %test13(bool %A, bool %B) {
	%C = setge bool %A, %B       ; A | ~B
	ret bool %C
}
bool %test14(bool %A, bool %B) {
	%C = seteq bool %A, %B  ; ~(A ^ B)
	ret bool %C
}

bool %test16(uint %A) {
	%B = and uint %A, 5
	%C = seteq uint %B, 8    ; Is never true
	ret bool %C
}

bool %test17(ubyte %A) {
	%B = or ubyte %A, 1
	%C = seteq ubyte %B, 2   ; Always false
	ret bool %C
}

