; RUN: opt < %s -instcombine -S | FileCheck %s
; PR1570

define i1 @f(float %X, float %Y) {
entry:
        %tmp3 = fcmp olt float %X, %Y           ; <i1> [#uses=1]
        %toBoolnot5 = xor i1 %tmp3, true                ; <i1> [#uses=1]
        ret i1 %toBoolnot5
; CHECK-LABEL: @f(
; CHECK-NEXT: entry:
; CHECK-NEXT: %toBoolnot5 = fcmp uge float %X, %Y
; CHECK-NEXT: ret i1 %toBoolnot5
}
