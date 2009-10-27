; RUN: llc < %s -march=arm | FileCheck %s -check-prefix=CHECKV4
; RUN: llc < %s -march=arm -mattr=+v5t | FileCheck %s -check-prefix=CHECKV5
; RUN: llc < %s -march=arm -mtriple=arm-linux-gnueabi\
; RUN:   -relocation-model=pic | FileCheck %s -check-prefix=CHECKELF

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; CHECKV4: mov lr, pc
; CHECKV5: blx
; CHECKELF: PLT
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @g.upgrd.1() {
        %tmp = load i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = tail call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}
