; RUN: llvm-as < %s | opt -raise | llvm-dis | grep '= cast' | not grep \*

%FILE = type { int, ubyte*, ubyte*, ubyte, ubyte, uint, uint, uint }

uint %addfile(%FILE* %f) {
	%cast255 = cast %FILE* %f to sbyte*

	; Addreses a ubyte member in memory...
	%reg2421 = getelementptr sbyte* %cast255, long 24

	; Loads the ubyte
	%reg130 = load sbyte* %reg2421

	; Error, cast cannot convert the source operand to ubyte because then
	; the sign extension would not be performed.  Need to insert a cast.
	;
	%cast250 = cast sbyte %reg130 to uint  ; This is a sign extension instruction
	ret uint %cast250
}
