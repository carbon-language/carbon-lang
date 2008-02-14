; RUN: llvm-as < %s | opt -load-vn -gcse -instcombine | llvm-dis | grep sub

define i32 @test(i32* %P) {
	%X = volatile load i32* %P		; <i32> [#uses=1]
	%Y = volatile load i32* %P		; <i32> [#uses=1]
	%Z = sub i32 %X, %Y		; <i32> [#uses=1]
	ret i32 %Z
}
