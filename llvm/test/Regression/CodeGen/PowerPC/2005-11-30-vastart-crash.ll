; RUN: llvm-as < %s | llc
target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.2.0"
implementation   ; Functions:

void %bar(int %G, int %E, int %F, int %A, int %B, int %C, int %D, sbyte* %fmt, ...) {
	%ap = alloca sbyte*		; <sbyte**> [#uses=2]
	call void %llvm.va_start( sbyte** %ap )
	%tmp.1 = load sbyte** %ap		; <sbyte*> [#uses=1]
	%tmp.0 = call double %foo( sbyte* %tmp.1 )		; <double> [#uses=0]
	ret void
}

declare void %llvm.va_start(sbyte**)

declare double %foo(sbyte*)
