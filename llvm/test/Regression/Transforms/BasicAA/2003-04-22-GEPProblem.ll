; RUN: as < %s | opt -load-vn -gcse -instcombine | dis | grep sub

; BasicAA was incorrectly concluding that P1 and P2 didn't conflict!

int %test(int *%Ptr, long %V) {
	%P2 = getelementptr int* %Ptr, long 1
	%P1 = getelementptr int* %Ptr, long %V
	%X = load int* %P1
	store int 5, int* %P2

	%Y = load int* %P1

	%Z = sub int %X, %Y
	ret int %Z
}
