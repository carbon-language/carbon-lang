; RUN: opt < %s -instcombine -S | FileCheck %s
@.str = private unnamed_addr constant [14 x i8] c"hello world!\0A\00", align 1

; CHECK: @foo
; CHECK: printf
define void @foo() nounwind ssp uwtable "no-builtin" {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([14 x i8]* @.str, i32 0, i32 0))
  ret void
}

; CHECK: @bar
; CHECK: puts
define void @bar() nounwind ssp uwtable {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([14 x i8]* @.str, i32 0, i32 0))
  ret void
}

declare i32 @printf(i8*, ...)

attributes #0 = { nounwind ssp uwtable "no-builtin" }
attributes #1 = { nounwind ssp uwtable }
