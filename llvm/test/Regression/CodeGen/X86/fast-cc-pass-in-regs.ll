; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel -enable-x86-fastcc  | grep 'mov EDX, 1'
; check that fastcc is passing stuff in regs.

; Argument reg passing is disabled due to regalloc issues.  FIXME!
; XFAIL: *

declare fastcc long %callee(long)

long %caller() {
	%X = call fastcc long %callee(long 4294967299)  ;; (1ULL << 32) + 3
	ret long %X
}

fastcc long %caller2(long %X) {
	ret long %X
}
