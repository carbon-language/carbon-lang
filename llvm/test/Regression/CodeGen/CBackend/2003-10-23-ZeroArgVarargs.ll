; XFAIL: *
; RUN: llvm-as < %s | llc -march=c


declare sbyte* %llvm.va_start()
declare void %llvm.va_end(sbyte*)

void %test(...) {
	%P = call sbyte* %llvm.va_start()
	call void %llvm.va_end(sbyte* %P)
	ret void
}
