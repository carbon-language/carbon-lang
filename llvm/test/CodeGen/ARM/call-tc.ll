; RUN: llc < %s -mtriple=armv6-apple-darwin -mattr=+vfp2 -arm-tail-calls | FileCheck %s -check-prefix=CHECKV6
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -relocation-model=pic -mattr=+vfp2 -arm-tail-calls | FileCheck %s -check-prefix=CHECKELF
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -arm-tail-calls | FileCheck %s -check-prefix=CHECKT2

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @t1() {
; CHECKELF: t1:
; CHECKELF: bl g(PLT)
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @t2() {
; CHECKV6: t2:
; CHECKV6: bx r0 @ TAILCALL
        %tmp = load i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = tail call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}

define void @t3() {
; CHECKV6: t3:
; CHECKV6: b _t2  @ TAILCALL
; CHECKELF: t3:
; CHECKELF: b t2(PLT) @ TAILCALL
        tail call void @t2( )            ; <i32> [#uses=0]
        ret void
}

; Sibcall optimization of expanded libcalls. rdar://8707777
define double @t4(double %a) nounwind readonly ssp {
entry:
; CHECKV6: t4:
; CHECKV6: b _sin @ TAILCALL
; CHECKELF: t4:
; CHECKELF: b sin(PLT) @ TAILCALL
  %0 = tail call double @sin(double %a) nounwind readonly ; <double> [#uses=1]
  ret double %0
}

define float @t5(float %a) nounwind readonly ssp {
entry:
; CHECKV6: t5:
; CHECKV6: b _sinf @ TAILCALL
; CHECKELF: t5:
; CHECKELF: b sinf(PLT) @ TAILCALL
  %0 = tail call float @sinf(float %a) nounwind readonly ; <float> [#uses=1]
  ret float %0
}

declare float @sinf(float) nounwind readonly

declare double @sin(double) nounwind readonly

define i32 @t6(i32 %a, i32 %b) nounwind readnone {
entry:
; CHECKV6: t6:
; CHECKV6: b ___divsi3 @ TAILCALL
; CHECKELF: t6:
; CHECKELF: b __aeabi_idiv(PLT) @ TAILCALL
  %0 = sdiv i32 %a, %b
  ret i32 %0
}

; Make sure the tail call instruction isn't deleted
; rdar://8309338
declare void @foo() nounwind

define void @t7() nounwind {
entry:
; CHECKT2: t7:
; CHECKT2: blxeq _foo
; CHECKT2-NEXT: pop.w
; CHECKT2-NEXT: b.w _foo
  br i1 undef, label %bb, label %bb1.lr.ph

bb1.lr.ph:
  tail call void @foo() nounwind
  unreachable

bb:
  tail call void @foo() nounwind
  ret void
}
