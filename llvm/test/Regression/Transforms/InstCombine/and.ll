; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine | dis | grep and
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int %test1(int %A) {
	%B = and int %A, 0     ; zero result
	ret int %B
}

int %test2(int %A) {
	%B = and int %A, -1    ; noop
	ret int %B
}

bool %test3(bool %A) {
	%B = and bool %A, false  ; always = false
	ret bool %B
}

bool %test4(bool %A) {
	%B = and bool %A, true  ; noop
	ret bool %B
}

int %test5(int %A) {
	%B = and int %A, %A
	ret int %B
}

bool %test6(bool %A) {
	%B = and bool %A, %A
	ret bool %B
}

int %test7(int %A) {         ; A & ~A == 0
        %NotA = xor int %A, -1
        %B = and int %A, %NotA
        ret int %B
}

ubyte %test8(ubyte %A) {    ; AND associates
	%B = and ubyte %A, 3
	%C = and ubyte %B, 4
	ret ubyte %C
}

