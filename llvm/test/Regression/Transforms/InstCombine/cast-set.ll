; I'm not really sure if instcombine should do things like these.  LevelRaise 
; already sufficiently takes care of these cases, but level raise is really
; slow.  Might it be better to make there be an instcombine prepass before
; level raise that takes care of the obvious stuff?

; RUN: as < %s | opt -instcombine | dis | grep-not cast

bool %test1(int %X) {
	%A = cast int %X to uint
	%c = setne uint %A, 0        ; Convert to setne int %X, 0
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
	%C = shl uint %B, ubyte 1
	%D = cast uint %C to int
	ret int %D
}

