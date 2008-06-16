; RUN: bugpoint %s -dce -bugpoint-deletecalls -simplifycfg -silence-passes

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32" 
target triple = "i386-pc-linux-gnu"       

@.LC0 = internal global [13 x i8] c"Hello World\0A\00"          ; <[13 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
        call i32 (i8*, ...)* @printf( i8* getelementptr ([13 x i8]* @.LC0, i64 0, i64 0) )            ; <i32>:1 [#uses=0]
        ret i32 0
}
