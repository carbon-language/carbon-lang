; RUN: opt < %s -passes=globalopt -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@G = internal global i32* null          ; <i32**> [#uses=4]
; CHECK-NOT: global

define void @init() {
        %malloccall = tail call i8* @malloc(i64 4)      ; <i8*> [#uses=1]
        %P = bitcast i8* %malloccall to i32*            ; <i32*> [#uses=1]
        store i32* %P, i32** @G
        %GV = load i32*, i32** @G             ; <i32*> [#uses=1]
        store i32 0, i32* %GV
        ret void
}

declare noalias i8* @malloc(i64)

define i32 @get() {
        %GV = load i32*, i32** @G             ; <i32*> [#uses=1]
        %V = load i32, i32* %GV              ; <i32> [#uses=1]
        ret i32 %V
; CHECK: ret i32 0
}

define void @foo(i64 %Size) nounwind noinline #0 {
entry:
        %0 = load i32*, i32** @G, align 4
        ret void
}

attributes #0 = { null_pointer_is_valid }

