; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep alloca | grep '{'
implementation   ; Functions:

declare int %.callback_1(sbyte*)
declare void %.iter_2(int (sbyte*)*, sbyte*)

int %main() {
	%d = alloca { [80 x sbyte], int, uint }
	%tmp.0 = getelementptr { [80 x sbyte], int, uint }* %d, long 0, ubyte 2
	store uint 0, uint* %tmp.0
	%tmp.1 = getelementptr { [80 x sbyte], int, uint }* %d, long 0, ubyte 0, long 0
	call void %.iter_2( int (sbyte*)* %.callback_1, sbyte* %tmp.1 )
	ret int 0
}
