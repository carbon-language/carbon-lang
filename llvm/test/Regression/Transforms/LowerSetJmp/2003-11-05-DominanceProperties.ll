; RUN: llvm-as < %s | opt -lowersetjmp -disable-output
	%struct.jmpenv = type { int, sbyte }

implementation

declare void %Perl_sv_setpv()
declare int %llvm.setjmp(int *)

void %perl_call_sv() {
	call void %Perl_sv_setpv( )
	%tmp.335 = getelementptr %struct.jmpenv* null, long 0, ubyte 0
	%tmp.336 = call int %llvm.setjmp( int* null )
	store int %tmp.336, int* %tmp.335
	ret void
}

