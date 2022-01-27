; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-linux -mattr=+thumb2 | FileCheck %s

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; CHECK-LABEL: f:
; CHECK: bl {{_?}}g
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @h() {
; CHECK-LABEL: h:
; CHECK: blx r0
        %tmp = load i32 ()*, i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}
