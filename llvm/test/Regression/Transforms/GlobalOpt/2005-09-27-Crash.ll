; RUN: llvm-as < %s | opt -globalopt -disable-output
	%RPyString = type { int, %arraytype.Char }
	%arraytype.Char = type { int, [0 x sbyte] }
	%arraytype.Signed = type { int, [0 x int] }
	%functiontype.1 = type %RPyString* (int)
	%structtype.test = type { int, %arraytype.Signed }
%structinstance.test = internal global { int, { int, [2 x int] } } { int 41, { int, [2 x int] } { int 2, [2 x int] [ int 100, int 101 ] } }		; <{ int, { int, [2 x int] } }*> [#uses=1]

implementation   ; Functions:

fastcc void %pypy_array_constant() {
block0:
	%tmp.9 = getelementptr %structtype.test* cast ({ int, { int, [2 x int] } }* %structinstance.test to %structtype.test*), int 0, uint 0		; <int*> [#uses=0]
	ret void
}

fastcc void %new.varsizestruct.rpy_string() {
	unreachable
}

void %__entrypoint__pypy_array_constant() {
	call fastcc void %pypy_array_constant( )
	ret void
}

void %__entrypoint__raised_LLVMException() {
	ret void
}
