; Dominator set calculation is not calculating dominators for unreachable 
; blocks.  These blocks should at least dominate themselves.  This is 
; fouling up the verify pass.
;
; RUN: llvm-as < %s | opt -analyze -domset | grep BB

void %test() {
	ret void
BB:
	ret void
}
