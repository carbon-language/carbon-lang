; RUN: opt %s -instcombine -S | FileCheck %s
; PR5438

; TODO: This should also optimize down.
;define i32 @bar(i32 %a, i32 %b) nounwind readnone {
;entry:
;        %0 = icmp sgt i32 %a, -1        ; <i1> [#uses=1]
;        %1 = icmp slt i32 %b, 0         ; <i1> [#uses=1]
;        %2 = xor i1 %1, %0              ; <i1> [#uses=1]
;        %3 = zext i1 %2 to i32          ; <i32> [#uses=1]
;        ret i32 %3
;}

define i32 @qaz(i32 %a, i32 %b) nounwind readnone {
; CHECK: @qaz
entry:
; CHECK: xor i32 %a, %b
; CHECK; lshr i32 %0, 31
; CHECK: xor i32 %1, 1
        %0 = lshr i32 %a, 31            ; <i32> [#uses=1]
        %1 = lshr i32 %b, 31            ; <i32> [#uses=1]
        %2 = icmp eq i32 %0, %1         ; <i1> [#uses=1]
        %3 = zext i1 %2 to i32          ; <i32> [#uses=1]
        ret i32 %3
; CHECK-NOT: icmp
; CHECK-NOT: zext
; CHECK: ret i32 %2
}
