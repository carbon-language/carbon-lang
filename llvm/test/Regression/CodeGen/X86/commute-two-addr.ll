; The register allocator can commute two-address instructions to avoid
; insertion of register-register copies.

; Check that there are no register-register copies left.
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | not grep 'mov %E.X, %E.X'

%G = external global int

declare void %ext(int)

int %add_test(int %X, int %Y) {
	%Z = add int %X, %Y      ;; Last use of Y, but not of X.
	store int %Z, int* %G
	ret int %X
}

int %xor_test(int %X, int %Y) {
	%Z = xor int %X, %Y      ;; Last use of Y, but not of X.
	store int %Z, int* %G
	ret int %X
}
