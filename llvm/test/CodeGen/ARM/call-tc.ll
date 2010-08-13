; RUN: llc < %s -mtriple=arm-apple-darwin -march=arm | FileCheck %s -check-prefix=CHECKV4
; RUN: llc < %s -march=arm -mtriple=arm-apple-darwin -mattr=+v5t | FileCheck %s -check-prefix=CHECKV5
; RUN: llc < %s -march=arm -mtriple=arm-linux-gnueabi\
; RUN:   -relocation-model=pic | FileCheck %s -check-prefix=CHECKELF
; XFAIL: *

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @t1() {
; CHECKELF: t1:
; CHECKELF: PLT
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @t2() {
; CHECKV4: t2:
; CHECKV4: bx r0 @ TAILCALL
; CHECKV5: t2:
; CHECKV5: bx r0 @ TAILCALL
        %tmp = load i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = tail call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}

define i32* @t3(i32, i32, i32*, i32*, i32*) nounwind {
; CHECKV4: t3:
; CHECKV4: bx r{{.*}}
BB0:
  %5 = inttoptr i32 %0 to i32*                    ; <i32*> [#uses=1]
  %t35 = volatile load i32* %5                    ; <i32> [#uses=1]
  %6 = inttoptr i32 %t35 to i32**                 ; <i32**> [#uses=1]
  %7 = getelementptr i32** %6, i32 86             ; <i32**> [#uses=1]
  %8 = load i32** %7                              ; <i32*> [#uses=1]
  %9 = bitcast i32* %8 to i32* (i32, i32*, i32, i32*, i32*, i32*)* ; <i32* (i32, i32*, i32, i32*, i32*, i32*)*> [#uses=1]
  %10 = call i32* %9(i32 %0, i32* null, i32 %1, i32* %2, i32* %3, i32* %4) ; <i32*> [#uses=1]
  ret i32* %10
}

define void @t4() {
; CHECKV4: t4:
; CHECKV4: b _t2  @ TAILCALL
; CHECKV5: t4:
; CHECKV5: b _t2  @ TAILCALL
        tail call void @t2( )            ; <i32> [#uses=0]
        ret void
}
