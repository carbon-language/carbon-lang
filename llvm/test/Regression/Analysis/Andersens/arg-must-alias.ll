; RUN: llvm-as < %s | opt -anders-aa -load-vn -gcse -deadargelim | llvm-dis | not grep ARG

%G = internal constant int* null

implementation

internal int %internal(int* %ARG) {
	;; The 'Arg' argument must-aliases the null pointer, so it can be subsituted
	;; directly here, making it dead.
	store int* %ARG, int** %G
	ret int 0
}

int %foo() {
	%V = call int %internal(int* null)
	ret int %V
}
