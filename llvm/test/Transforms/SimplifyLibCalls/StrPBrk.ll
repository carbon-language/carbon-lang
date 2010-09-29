; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

target datalayout = "-p:64:64:64"

@hello = constant [12 x i8] c"hello world\00"
@w = constant [2 x i8] c"w\00"
@null = constant [1 x i8] zeroinitializer

declare i8* @strpbrk(i8*, i8*)

define void @test(i8* %s1, i8* %s2) {
	%hello_p = getelementptr [12 x i8]* @hello, i32 0, i32 0
	%w_p = getelementptr [2 x i8]* @w, i32 0, i32 0
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0
	%test1 = call i8* @strpbrk(i8* %null_p, i8* %s2)
	%test2 = call i8* @strpbrk(i8* %s1, i8* %null_p)
; CHECK-NOT: call i8* @strpbrk
	%test3 = call i8* @strpbrk(i8* %s1, i8* %w_p)
; CHECK: call i8* @strchr(i8* %s1, i32 119)
	%test4 = call i8* @strpbrk(i8* %hello_p, i8* %w_p)
; CHECK: getelementptr i8* %hello_p, i64 6
	%test5 = call i8* @strpbrk(i8* %s1, i8* %s2)
; CHECK: call i8* @strpbrk(i8* %s1, i8* %s2)
	ret void
}
