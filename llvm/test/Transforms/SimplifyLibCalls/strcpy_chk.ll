; RUN: opt < %s -simplify-libcalls -S | FileCheck %s
@a = common global [60 x i8] zeroinitializer, align 1 ; <[60 x i8]*> [#uses=1]
@.str = private constant [8 x i8] c"abcdefg\00"   ; <[8 x i8]*> [#uses=1]

define i8* @foo() nounwind {
; CHECK: @foo
; CHECK-NEXT: call i8* @strcpy
  %call = call i8* @__strcpy_chk(i8* getelementptr inbounds ([60 x i8]* @a, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i32 60) ; <i8*> [#uses=1]
  ret i8* %call
}

declare i8* @__strcpy_chk(i8*, i8*, i32) nounwind
