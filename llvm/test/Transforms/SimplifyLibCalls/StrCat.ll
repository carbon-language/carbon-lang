; Test that the StrCatOptimizer works correctly
; PR3661
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*strcat}
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   grep {puts.*%arg1}

; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "-p:64:64:64"

@hello = constant [6 x i8] c"hello\00"		; <[6 x i8]*> [#uses=1]
@null = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]
@null_hello = constant [7 x i8] c"\00hello\00"		; <[7 x i8]*> [#uses=1]

declare i8* @strcat(i8*, i8*)

declare i32 @puts(i8*)

define i32 @main() {
	%target = alloca [1024 x i8]		; <[1024 x i8]*> [#uses=1]
	%arg1 = getelementptr [1024 x i8]* %target, i32 0, i32 0		; <i8*> [#uses=2]
	store i8 0, i8* %arg1
	%arg2 = getelementptr [6 x i8]* @hello, i32 0, i32 0		; <i8*> [#uses=1]
	%rslt1 = call i8* @strcat( i8* %arg1, i8* %arg2 )		; <i8*> [#uses=1]
	%arg3 = getelementptr [1 x i8]* @null, i32 0, i32 0		; <i8*> [#uses=1]
	%rslt2 = call i8* @strcat( i8* %rslt1, i8* %arg3 )		; <i8*> [#uses=1]
	%arg4 = getelementptr [7 x i8]* @null_hello, i32 0, i32 0		; <i8*> [#uses=1]
	%rslt3 = call i8* @strcat( i8* %rslt2, i8* %arg4 )		; <i8*> [#uses=1]
	call i32 @puts( i8* %rslt3 )		; <i32>:1 [#uses=0]
	ret i32 0
}

