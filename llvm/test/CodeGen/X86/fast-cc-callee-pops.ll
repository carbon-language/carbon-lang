; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=x86 -x86-asm-syntax=intel -mcpu=yonah | grep {ret 20}

; Check that a fastcc function pops its stack variables before returning.

x86_fastcallcc void %func(long %X, long %Y, float %G, double %Z) {
	ret void
}
