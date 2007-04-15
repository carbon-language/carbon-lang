; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep {add ESP, 8}

target triple = "i686-pc-linux-gnu"

declare x86_fastcallcc void %func(int *%X, long %Y)

x86_fastcallcc void %caller(int, long) {
	%X = alloca int
	call x86_fastcallcc void %func(int* %X, long 0)   ;; not a tail call
	ret void
}
