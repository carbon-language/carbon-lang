; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep sub

declare i32 @strlen(i8*) readonly

define i32 @test(i8* %P) {
	%X = call i32 @strlen( i8* %P ) readonly		; <i32> [#uses=2]
	%A = add i32 %X, 14		; <i32> [#uses=1]
	%Y = call i32 @strlen( i8* %P ) readonly		; <i32> [#uses=1]
	%Z = sub i32 %X, %Y		; <i32> [#uses=1]
	%B = add i32 %A, %Z		; <i32> [#uses=1]
	ret i32 %B
}
