; Induction variable pass is doing bad things with pointer induction vars, 
; trying to do arithmetic on them directly.
;
; RUN: llvm-as < %s | opt -indvars
;
void %test(int %A, uint %S, sbyte* %S) {

	br label %Loop
Loop:
	%PIV = phi sbyte* [%S, %0], [%PIVNext, %Loop]

	%PIV = cast sbyte* %PIV to ulong
	%PIVNext = add ulong %PIV, 8
	%PIVNext = cast ulong %PIVNext to sbyte*
	br label %Loop
}
