; I'm not really sure if instcombine should do things like these.  LevelRaise 
; already sufficiently takes care of these cases, but level raise is really
; slow.  Might it be better to make there be an instcombine prepass before
; level raise that takes care of the obvious stuff?

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep cast

bool %test1(int %X) {
	%A = cast int %X to uint
	%c = setne uint %A, 12        ; Convert to setne int %X, 12
	ret bool %c
}

bool %test2(int %X, int %Y) {
	%A = cast int %X to uint
	%B = cast int %Y to uint
	%c = setne uint %A, %B       ; Convert to setne int %X, %Y
	ret bool %c
}

bool %test3(int %A, int %B) {
        %cond216 = setlt int %A, %B             ; <bool> [#uses=1]
        %cst109 = cast bool %cond216 to uint           ; <uint> [#uses=1]
        %cond219 = setgt int %A, %B             ; <bool> [#uses=1]
        %cst111 = cast bool %cond219 to uint           ; <uint> [#uses=1]
        %reg113 = and uint %cst109, %cst111           ; <uint> [#uses=1]
        %cst222 = cast uint %reg113 to bool             ; <int> [#uses=1]
        ret bool %cst222
}

int %test4(int %A) {
	%B = cast int %A to uint
	%C = shl uint %B, ubyte 2
	%D = cast uint %C to int
	ret int %D
}

short %test5(short %A) {
	%B = cast short %A to uint
	%C = and uint %B, 15
	%D = cast uint %C to short
	ret short %D
}

bool %test6(bool %A) {
	%B = cast bool %A to int
	%C = setne int %B, 0
	ret bool %C
}

bool %test6a(bool %A) {
	%B = cast bool %A to int
	%C = setne int %B, -1    ; Always true!
	ret bool %C
}

bool %test7(sbyte* %A) {
	%B = cast sbyte* %A to int*
	%C = seteq int* %B, null
	ret bool %C
}
