;; X's live range extends beyond the shift, so the register allocator
;; cannot coalesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.

; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | not grep 'mov %E.X, %E.X'

; FIXME: We need live variable information about flags to do this xform safely. :(
; XFAIL: *

%G = external global int

int %test1(int %X, int %Y) {
	%Z = add int %X, %Y
	volatile store int %Y, int* %G
	volatile store int %Z, int* %G
	ret int %X
}

int %test2(int %X) {
	%Z = add int %X, 1  ;; inc
	volatile store int %Z, int* %G
	ret int %X
}
