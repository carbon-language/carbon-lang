; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o - | FileCheck %s -check-prefix=V6
; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s -check-prefix=V4
; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m3 %s -o - | FileCheck %s -check-prefix=M3

define i32 @smulhi(i32 %x, i32 %y) nounwind {
; V6-LABEL: smulhi:
; V6: smmul

; V4-LABEL: smulhi:
; V4: smull

; M3-LABEL: smulhi:
; M3: smull
        %tmp = sext i32 %x to i64               ; <i64> [#uses=1]
        %tmp1 = sext i32 %y to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        %tmp3 = lshr i64 %tmp2, 32              ; <i64> [#uses=1]
        %tmp3.upgrd.1 = trunc i64 %tmp3 to i32          ; <i32> [#uses=1]
        ret i32 %tmp3.upgrd.1
}

define i32 @umulhi(i32 %x, i32 %y) nounwind {
; V6-LABEL: umulhi:
; V6: umull

; V4-LABEL: umulhi:
; V4: umull

; M3-LABEL: umulhi:
; M3: umull
        %tmp = zext i32 %x to i64               ; <i64> [#uses=1]
        %tmp1 = zext i32 %y to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        %tmp3 = lshr i64 %tmp2, 32              ; <i64> [#uses=1]
        %tmp3.upgrd.2 = trunc i64 %tmp3 to i32          ; <i32> [#uses=1]
        ret i32 %tmp3.upgrd.2
}

; rdar://r10152911
define i32 @t3(i32 %a) nounwind {
; V6-LABEL: t3:
; V6: smmla

; V4-LABEL: t3:
; V4: smull

; M3-LABEL: t3:
; M3-NOT: smmla
; M3: smull
entry:
  %tmp1 = mul nsw i32 %a, 3
  %tmp2 = sdiv i32 %tmp1, 23
  ret i32 %tmp2
}
