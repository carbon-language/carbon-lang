; RUN: llvm-as < %s | llc -march=x86
; PR825

long %test() {
	%tmp.i5 = call long asm sideeffect "rdtsc", "=A,~{dirflag},~{fpsr},~{flags}"( )		; <long> [#uses=0]
	ret long %tmp.i5
}
