; RUN: llvm-as < %s | opt -raiseallocs -disable-output
implementation   ; Functions:

void %main() {
	%tmp.13 = call int (...)* %free( int 32 )
	%tmp.14 = cast int %tmp.13 to int*
	ret void
}

declare int %free(...)
