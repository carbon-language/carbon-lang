; I'm not really sure if instcombine should do things like these.  LevelRaise 
; already sufficiently takes care of these cases, but level raise is really
; slow.  Might it be better to make there be an instcombine prepass before
; level raise that takes care of the obvious stuff?

; RUN: if as < %s | opt -instcombine | dis | grep cast
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

bool "test1"(int %X) {
	%A = cast int %X to uint
	%c = setne uint %A, 0        ; Convert to setne int %X, 0
	ret bool %c
}

bool "test2"(int %X, int %Y) {
	%A = cast int %X to uint
	%B = cast int %Y to uint
	%c = setne uint %A, %B       ; Convert to setne int %X, %Y
	ret bool %c
}

