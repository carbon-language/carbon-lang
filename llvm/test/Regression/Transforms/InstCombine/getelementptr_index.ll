; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep cast

target endian = little
target pointersize = 32

int* %test(int* %X, long %Idx) {
	; Should insert a cast to int on this target
	%R = getelementptr int* %X, long %Idx
	ret int* %R
}
