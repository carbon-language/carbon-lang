; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

implementation
declare sbyte* %llvm.va_start()
declare sbyte* %llvm.va_copy(sbyte*)
declare void %llvm.va_end(sbyte*)

int %test(int %X, ...) {
	%ap = call sbyte* %llvm.va_start()
	%aq = call sbyte* %llvm.va_copy(sbyte* %ap)
	call void %llvm.va_end(sbyte* %aq)
	
	%tmp = vaarg sbyte* %ap, int 
	%ap2 = vanext sbyte* %ap, int

	call void %llvm.va_end(sbyte* %ap2)
	ret int %tmp
}
