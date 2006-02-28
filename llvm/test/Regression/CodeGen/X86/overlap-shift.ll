;; X's live range extends beyond the shift, so the register allocator
;; cannot coallesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.

; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | not grep 'mov %E.X, %E.X'

%G = external global int

int %test1(int %X) {
	%Z = shl int %X, ubyte 2
	volatile store int %Z, int* %G
	ret int %X
}
