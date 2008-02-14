; RUN: llvm-as < %s | opt -anders-aa -load-vn -gcse -deadargelim | llvm-dis | not grep ARG

@G = internal constant i32* null

define internal i32 @internal(i32* %ARG) {
	;; The 'Arg' argument must-aliases the null pointer, so it can be subsituted
	;; directly here, making it dead.
	store i32* %ARG, i32** @G
	ret i32 0
}

define i32 @foo() {
	%V = call i32 @internal(i32* null)
	ret i32 %V
}
