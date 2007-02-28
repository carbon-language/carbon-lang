; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel -enable-x86-fastcc -mcpu=yonah | grep 'ret 28'

; Check that a fastcc function pops its stack variables before returning.

x86_fastcallcc void %func(long %X, long %Y, float %G, double %Z) {
	ret void
}
