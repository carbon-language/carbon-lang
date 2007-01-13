; XFAIL: *
; RUN: llvm-as < %s | llc -march=c


declare i8* %llvm.va_start()
declare void %llvm.va_end(i8*)

void %test(...) {
	%P = call i8* %llvm.va_start()
	call void %llvm.va_end(i8* %P)
	ret void
}
