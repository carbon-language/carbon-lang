target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; RUN: opt < %s -instcombine -S | FileCheck %s
; END.

declare void @use(...)

; Zero byte allocas should be deleted.
; CHECK: @test
; CHECK-NOT: alloca
define void @test() {
        %X = alloca [0 x i32]           ; <[0 x i32]*> [#uses=1]
        call void (...)* @use( [0 x i32]* %X )
        %Y = alloca i32, i32 0          ; <i32*> [#uses=1]
        call void (...)* @use( i32* %Y )
        %Z = alloca {  }                ; <{  }*> [#uses=1]
        call void (...)* @use( {  }* %Z )
        ret void
}

; Zero byte allocas should be deleted.
; CHECK: @test2
; CHECK-NOT: alloca
define void @test2() {
        %A = alloca i32         ; <i32*> [#uses=1]
        store i32 123, i32* %A
        ret void
}

; Zero byte allocas should be deleted.
; CHECK: @test3
; CHECK-NOT: alloca
define void @test3() {
        %A = alloca { i32 }             ; <{ i32 }*> [#uses=1]
        %B = getelementptr { i32 }* %A, i32 0, i32 0            ; <i32*> [#uses=1]
        store i32 123, i32* %B
        ret void
}

; CHECK: @test4
; CHECK: = zext i32 %n to i64
; CHECK: %A = alloca i32, i64 %
define i32* @test4(i32 %n) {
  %A = alloca i32, i32 %n
  ret i32* %A
}
