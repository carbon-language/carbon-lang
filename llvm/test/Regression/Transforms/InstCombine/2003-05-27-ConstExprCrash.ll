; RUN: as < %s | opt -instcombine -disable-output

%X = global int 5
long %test() {
        %C = add long 1, 2
	%V = add long cast(int* %X to long), %C
	ret long %V
}
