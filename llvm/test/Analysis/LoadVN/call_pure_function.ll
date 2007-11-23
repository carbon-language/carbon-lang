; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep sub

declare i32 @strlen(i8*) readonly

declare void @use(i32)

define i8 @test(i8* %P, i8* %Q) {
	%A = load i8* %Q		; <i8> [#uses=1]
	%X = call i32 @strlen( i8* %P ) readonly		; <i32> [#uses=1]
	%B = load i8* %Q		; <i8> [#uses=1]
	call void @use( i32 %X )
	%C = sub i8 %A, %B		; <i8> [#uses=1]
	ret i8 %C
}
