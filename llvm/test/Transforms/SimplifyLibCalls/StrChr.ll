; Test that the StrChrOptimizer works correctly
; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "-p:64:64:64"

@hello = constant [14 x i8] c"hello world\5Cn\00"
@null = constant [1 x i8] zeroinitializer

declare i8* @strchr(i8*, i32)

define i32 @foo(i32 %index) {
	%hello_p = getelementptr [14 x i8]* @hello, i32 0, i32 0
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0
	%world = call i8* @strchr(i8* %hello_p, i32 119)
; CHECK: getelementptr i8* %hello_p, i64 6
	%ignore = call i8* @strchr(i8* %null_p, i32 119)
; CHECK-NOT: call i8* strchr
	%null = call i8* @strchr(i8* %hello_p, i32 0)
; CHECK: getelementptr i8* %hello_p, i64 13
	%result = call i8* @strchr(i8* %hello_p, i32 %index)
; CHECK: call i8* @memchr(i8* %hello_p, i32 %index, i64 14)
	ret i32 %index
}

