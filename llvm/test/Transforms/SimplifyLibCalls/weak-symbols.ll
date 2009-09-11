; RUN: opt < %s -simplify-libcalls -S | FileCheck %s
; PR4738

; SimplifyLibcalls shouldn't assume anything about weak symbols.

@real_init = weak_odr constant [2 x i8] c"y\00"
@fake_init = weak constant [2 x i8] c"y\00"
@.str = private constant [2 x i8] c"y\00"

; CHECK: define i32 @foo
; CHECK: call i32 @strcmp
define i32 @foo() nounwind {
entry:
  %t0 = call i32 @strcmp(i8* getelementptr inbounds ([2 x i8]* @fake_init, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8]* @.str, i64 0, i64 0)) nounwind readonly
  ret i32 %t0
}

; CHECK: define i32 @bar
; CHECK: ret i32 0
define i32 @bar() nounwind {
entry:
  %t0 = call i32 @strcmp(i8* getelementptr inbounds ([2 x i8]* @real_init, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8]* @.str, i64 0, i64 0)) nounwind readonly
  ret i32 %t0
}

declare i32 @strcmp(i8*, i8*) nounwind readonly
