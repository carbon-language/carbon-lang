; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

implementation
declare void %llvm.va_start(sbyte**, ...)
declare void %llvm.va_copy(sbyte**, sbyte*)
declare void %llvm.va_end(sbyte**)

int %test(int %X, ...) {
	%ap = alloca sbyte*
	%aq = alloca sbyte*
	call void (sbyte**, ...)* %llvm.va_start(sbyte** %ap, int %X)
	%apv = load sbyte** %ap
	call void %llvm.va_copy(sbyte** %aq, sbyte* %apv)
	call void %llvm.va_end(sbyte** %aq)
	
	%tmp = va_arg sbyte** %ap, int 

	call void %llvm.va_end(sbyte** %ap)
	ret int %tmp
}
