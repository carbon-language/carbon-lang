; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

target datalayout = "-p:64:64:64"

@abcba = constant [6 x i8] c"abcba\00"
@abc = constant [4 x i8] c"abc\00"
@null = constant [1 x i8] zeroinitializer

declare i64 @strspn(i8*, i8*)

define i64 @testspn(i8* %s1, i8* %s2) {
  	%abcba_p = getelementptr [6 x i8]* @abcba, i32 0, i32 0
	%abc_p = getelementptr [4 x i8]* @abc, i32 0, i32 0
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0
	%test1 = call i64 @strspn(i8* %s1, i8* %null_p)
	%test2 = call i64 @strspn(i8* %null_p, i8* %s2)
	%test3 = call i64 @strspn(i8* %abcba_p, i8* %abc_p)
; CHECK-NOT: call i64 @strspn
	%test4 = call i64 @strspn(i8* %s1, i8* %s2)
; CHECK: call i64 @strspn(i8* %s1, i8* %s2)
	ret i64 %test3
; CHECK ret i64 5
}

declare i64 @strcspn(i8*, i8*)

define i64 @testcspn(i8* %s1, i8* %s2) {
  	%abcba_p = getelementptr [6 x i8]* @abcba, i32 0, i32 0
	%abc_p = getelementptr [4 x i8]* @abc, i32 0, i32 0
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0
	%test1 = call i64 @strcspn(i8* %s1, i8* %null_p)
; CHECK: call i64 @strlen(i8* %s1)
	%test2 = call i64 @strcspn(i8* %null_p, i8* %s2)
	%test3 = call i64 @strcspn(i8* %abcba_p, i8* %abc_p)
; CHECK-NOT: call i64 @strcspn
	%test4 = call i64 @strcspn(i8* %s1, i8* %s2)
; CHECK: call i64 @strcspn(i8* %s1, i8* %s2)
        %add0 = add i64 %test1, %test3
; CHECK: add i64 %{{.+}}, 0
	ret i64 %add0
}
