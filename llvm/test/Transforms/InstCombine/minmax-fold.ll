; RUN: opt -S -instcombine < %s | FileCheck %s

; CHECK-LABEL: @t1
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: sext
define i64 @t1(i32 %a) {
  ; This is the canonical form for a type-changing min/max.
  %1 = icmp slt i32 %a, 5
  %2 = select i1 %1, i32 %a, i32 5
  %3 = sext i32 %2 to i64
  ret i64 %3
}

; CHECK-LABEL: @t2
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: sext
define i64 @t2(i32 %a) {
  ; Check this is converted into canonical form, as above.
  %1 = icmp slt i32 %a, 5
  %2 = sext i32 %a to i64
  %3 = select i1 %1, i64 %2, i64 5
  ret i64 %3
}

; CHECK-LABEL: @t3
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: zext
define i64 @t3(i32 %a) {
  ; Same as @t2, with flipped operands and zext instead of sext.
  %1 = icmp ult i32 %a, 5
  %2 = zext i32 %a to i64
  %3 = select i1 %1, i64 5, i64 %2
  ret i64 %3
}

; CHECK-LABEL: @t4
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: trunc
define i32 @t4(i64 %a) {
  ; Same again, with trunc.
  %1 = icmp slt i64 %a, 5
  %2 = trunc i64 %a to i32
  %3 = select i1 %1, i32 %2, i32 5
  ret i32 %3
}

; CHECK-LABEL: @t5
; CHECK-NEXT: icmp
; CHECK-NEXT: zext
; CHECK-NEXT: select
define i64 @t5(i32 %a) {
  ; Same as @t3, but with mismatched signedness between icmp and zext.
  ; InstCombine should leave this alone.
  %1 = icmp slt i32 %a, 5
  %2 = zext i32 %a to i64
  %3 = select i1 %1, i64 5, i64 %2
  ret i64 %3
}

; CHECK-LABEL: @t6
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: sitofp
define float @t6(i32 %a) {
  %1 = icmp slt i32 %a, 0
  %2 = select i1 %1, i32 %a, i32 0
  %3 = sitofp i32 %2 to float
  ret float %3
}

; CHECK-LABEL: @t7
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: trunc
define i16 @t7(i32 %a) {
  %1 = icmp slt i32 %a, -32768
  %2 = trunc i32 %a to i16
  %3 = select i1 %1, i16 %2, i16 -32768
  ret i16 %3
}
