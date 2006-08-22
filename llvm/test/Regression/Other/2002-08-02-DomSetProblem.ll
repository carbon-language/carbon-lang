; Dominator set calculation is not calculating dominators for unreachable 
; blocks.  These blocks should at least dominate themselves.  This is 
; fouling up the verify pass.
;
; RUN: opt -analyze -domset %s | grep BB

void %test() {
	ret void
BB:
	ret void
}
