; Linking these two translation units causes there to be two LLVM values in the
; symbol table with the same name and same type.  When this occurs, the symbol
; table class is DROPPING one of the values, instead of renaming it like a nice
; little symbol table.  This is causing llvm-link to die, at no fault of its
; own.

; RUN: llvm-as < %s > %t.out2.bc
; RUN: echo "%T1 = type opaque  %GVar = external global %T1*" | llvm-as > %t.out1.bc
; RUN: llvm-link %t.out[12].bc

	%T1 = type opaque
	%T2 = type int

%GVar = global %T2 * null

implementation

void %foo(%T2 * %X) {
	%X = cast %T2* %X to %T1 *
	ret void
}


