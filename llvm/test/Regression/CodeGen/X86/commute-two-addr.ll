; The register allocator can commute two-address instructions to avoid
; insertion of register-register copies.

; Make sure there are only 3 mov's for each testcase
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | grep 'mov ' | wc -l | grep 6


target triple = "i686-pc-linux-gnu"

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
