; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 -mtriple=armv6-apple-darwin10 | FileCheck %s

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; CHECK: f:
; CHECK: blx
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @h() {
; CHECK: h:
; CHECK: blx r0
        %tmp = load i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = tail call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}
