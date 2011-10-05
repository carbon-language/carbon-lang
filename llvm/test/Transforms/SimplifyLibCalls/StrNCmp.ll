; Test that the StrCmpOptimizer works correctly
; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"		; <[6 x i8]*> [#uses=1]
@hell = constant [5 x i8] c"hell\00"		; <[5 x i8]*> [#uses=1]
@bell = constant [5 x i8] c"bell\00"		; <[5 x i8]*> [#uses=1]
@null = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]

declare i32 @strncmp(i8*, i8*, i32)

; strcmp("", x) -> -*x
define i32 @test1(i8* %str) {
  %temp1 = call i32 @strncmp(i8* getelementptr inbounds ([1 x i8]* @null, i32 0, i32 0), i8* %str, i32 10)
  ret i32 %temp1
  ; CHECK: @test1
  ; CHECK: %strcmpload = load i8* %str
  ; CHECK: %1 = zext i8 %strcmpload to i32
  ; CHECK: %temp1 = sub i32 0, %1
  ; CHECK: ret i32 %temp1
}

; strcmp(x, "") -> *x
define i32 @test2(i8* %str) {
  %temp1 = call i32 @strncmp(i8* %str, i8* getelementptr inbounds ([1 x i8]* @null, i32 0, i32 0), i32 10)
  ret i32 %temp1
  ; CHECK: @test2
  ; CHECK: %strcmpload = load i8* %str
  ; CHECK: %temp1 = zext i8 %strcmpload to i32
  ; CHECK: ret i32 %temp1
}

; strncmp(x, y, n)  -> cnst
define i32 @test3() {
  %temp1 = call i32 @strncmp(i8* getelementptr inbounds ([5 x i8]* @hell, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @hello, i32 0, i32 0), i32 10)
  ret i32 %temp1
  ; CHECK: @test3
  ; CHECK: ret i32 -1
}
define i32 @test4() {
  %temp1 = call i32 @strncmp(i8* getelementptr inbounds ([5 x i8]* @hell, i32 0, i32 0), i8* getelementptr inbounds ([1 x i8]* @null, i32 0, i32 0), i32 10)
  ret i32 %temp1
  ; CHECK: @test4
  ; CHECK: ret i32 1
}
define i32 @test5() {
  %temp1 = call i32 @strncmp(i8* getelementptr inbounds ([5 x i8]* @hell, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @hello, i32 0, i32 0), i32 4)
  ret i32 %temp1
  ; CHECK: @test5
  ; CHECK: ret i32 0
}

; strncmp(x,y,1) -> memcmp(x,y,1)
define i32 @test6(i8* %str1, i8* %str2) {
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 1)
  ret i32 %temp1
  ; CHECK: @test6
  ; CHECK: load i8*
  ; CHECK: load i8*
  ; CHECK: sub i32
}

; strncmp(x,y,0)   -> 0
define i32 @test7(i8* %str1, i8* %str2) {
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 0)
  ret i32 %temp1
  ; CHECK: @test7
  ; CHECK: ret i32 0
}

; strncmp(x,x,n)  -> 0
define i32 @test8(i8* %str, i32 %n) {
  %temp1 = call i32 @strncmp(i8* %str, i8* %str, i32 %n)
  ret i32 %temp1
  ; CHECK: @test8
  ; CHECK: ret i32 0
}
