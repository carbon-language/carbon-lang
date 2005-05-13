; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel -enable-x86-fastcc  | grep 'ret 20'

; Check that a fastcc function pops its stack variables before returning.

fastcc void %func(long %X, long %Y, float %G, double %Z) {
	ret void
}
