; RUN: llvm-as < %s | opt -instcombine -disable-output

%Ty = type opaque

int %test(%Ty *%X) {
	%Y = cast %Ty* %X to int*
	%Z = load int* %Y
	ret int %Z
}
