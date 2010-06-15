; RUN: opt < %s -simplify-libcalls -S | FileCheck %s
; PR5783

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.0"

@.str = private constant [1 x i8] zeroinitializer ; <[1 x i8]*> [#uses=1]
@.str1 = private constant [2 x i8] c"a\00"        ; <[2 x i8]*> [#uses=1]
@.str2 = private constant [6 x i8] c"abcde\00"    ; <[6 x i8]*> [#uses=1]
@.str3 = private constant [4 x i8] c"bcd\00"      ; <[4 x i8]*> [#uses=1]

define i8* @test1(i8* %P) nounwind readonly {
entry:
  %call = tail call i8* @strstr(i8* %P, i8* getelementptr inbounds ([1 x i8]* @.str, i32 0, i32 0)) nounwind ; <i8*> [#uses=1]
  ret i8* %call
; strstr(P, "") -> P
; CHECK: @test1
; CHECK: ret i8* %P
}

declare i8* @strstr(i8*, i8* nocapture) nounwind readonly

define i8* @test2(i8* %P) nounwind readonly {
entry:
  %call = tail call i8* @strstr(i8* %P, i8* getelementptr inbounds ([2 x i8]* @.str1, i32 0, i32 0)) nounwind ; <i8*> [#uses=1]
  ret i8* %call
; strstr(P, "a") -> strchr(P, 'a')
; CHECK: @test2
; CHECK: @strchr(i8* %P, i32 97)
}

define i8* @test3(i8* nocapture %P) nounwind readonly {
entry:
  %call = tail call i8* @strstr(i8* getelementptr inbounds ([6 x i8]* @.str2, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8]* @.str3, i32 0, i32 0)) nounwind ; <i8*> [#uses=1]
  ret i8* %call
; strstr("abcde", "bcd") -> "abcde"+1
; CHECK: @test3
; CHECK: getelementptr inbounds ([6 x i8]* @.str2, i32 0, i64 1)
}

define i8* @test4(i8* %P) nounwind readonly {
entry:
  %call = tail call i8* @strstr(i8* %P, i8* %P) nounwind ; <i8*> [#uses=1]
  ret i8* %call
; strstr(P, P) -> P
; CHECK: @test4
; CHECK: ret i8* %P
}

define i1 @test5(i8* %P, i8* %Q) nounwind readonly {
entry:
  %call = tail call i8* @strstr(i8* %P, i8* %Q) nounwind ; <i8*> [#uses=1]
  %cmp = icmp eq i8* %call, %P
  ret i1 %cmp
; CHECK: @test5
; CHECK: [[LEN:%[a-z]+]] = call {{i[0-9]+}} @strlen(i8* %Q)
; CHECK: [[NCMP:%[a-z]+]] = call {{i[0-9]+}} @strncmp(i8* %P, i8* %Q, {{i[0-9]+}} [[LEN]])
; CHECK: icmp eq {{i[0-9]+}} [[NCMP]], 0
; CHECK: ret i1
}
