; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel -enable-x86-fastcc  | grep 'add %ESP, 8'

declare fastcc void %func(int *%X, long %Y)

fastcc void %caller(int, long) {
	%X = alloca int
	call fastcc void %func(int* %X, long 0)   ;; not a tail call
	ret void
}
