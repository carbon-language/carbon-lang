; RUN: if as < %s | opt -raise | dis | grep '= cast' | grep \*
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%FILE = type { int, ubyte*, ubyte*, ubyte, ubyte, uint, uint, uint }

uint %addfile(%FILE* %f) {
	%cast255 = cast %FILE* %f to sbyte*		; <sbyte*> [#uses=1]
	%reg2421 = getelementptr sbyte* %cast255, uint 24		; <sbyte*> [#uses=1]
	%reg130 = load sbyte* %reg2421		; <sbyte> [#uses=1]
	%cast250 = cast sbyte %reg130 to uint		; <uint> [#uses=1]
	ret uint %cast250
}
