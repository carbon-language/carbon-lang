; RUN: llvm-as < %s | opt -indvars  | llvm-dis | grep -C1 Loop: | grep %indvar

; The indvar simplification code should ensure that the first PHI in the block 
; is the canonical one!

int %test() {
	br label %Loop
Loop:
	%NonIndvar = phi int [200, %0], [%NonIndvarNext, %Loop]
	%Canonical = phi int [0, %0], [%CanonicalNext, %Loop]

	%NonIndvarNext = div int %NonIndvar, 2
	%CanonicalNext = add int %Canonical, 1
	br label %Loop
}

