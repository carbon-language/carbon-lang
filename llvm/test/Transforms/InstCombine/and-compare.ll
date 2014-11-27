; RUN: opt < %s -instcombine -S | \
; RUN: FileCheck %s

; Should be optimized to one and.
define i1 @test1(i32 %a, i32 %b) {
; CHECK-LABEL: @test1(
; CHECK-NEXT: %1 = xor i32 %a, %b
; CHECK-NEXT: %2 = and i32 %1, 65280
; CHECK-NEXT: %tmp = icmp ne i32 %2, 0
; CHECK-NEXT: ret i1 %tmp
        %tmp1 = and i32 %a, 65280               ; <i32> [#uses=1]
        %tmp3 = and i32 %b, 65280               ; <i32> [#uses=1]
        %tmp = icmp ne i32 %tmp1, %tmp3         ; <i1> [#uses=1]
        ret i1 %tmp
}
