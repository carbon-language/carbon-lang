; RUN: llvm-as < %s | opt -inline -disable-output

implementation   ; Functions:

int %reload() {
reloadentry:
	br label %A
A:
	call void %callee( )
	ret int 0
}

internal void %callee() {
entry:
	%X = alloca sbyte, uint 0
	%Y = cast int 0 to uint
	%Z = alloca sbyte, uint %Y
	ret void
}
