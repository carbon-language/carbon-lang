; RUN: opt -S -instsimplify < %s | FileCheck %s

; CHECK-LABEL: @one
define i32 @one(i32 %a) {
; CHECK: ret i32 0
  %b = icmp ugt i32 %a, 5
  %c = select i1 %b, i32 2, i32 %a
  %d = lshr i32 %c, 24
  ret i32 %d
}

; CHECK-LABEL: @two
define i32 @two(i32 %a) {
; CHECK: ret i32 0
  %x = add nsw i32 %a, 4
  %b = icmp ugt i32 %x, 5
  %c = select i1 %b, i32 2, i32 %a
  %d = lshr i32 %c, 24
  ret i32 %d
}

; CHECK-LABEL: @two_no_nsw
define i32 @two_no_nsw(i32 %a) {
; CHECK: ret i32 %d
  %x = add i32 %a, 4
  %b = icmp ugt i32 %x, 5
  %c = select i1 %b, i32 2, i32 %a
  %d = lshr i32 %c, 24
  ret i32 %d
}

; CHECK-LABEL: @three
define i32 @three(i32 %a) {
; CHECK: ret i32 0
  %x = add nsw i32 %a, -4
  %b = icmp ugt i32 %a, 5
  %c = select i1 %b, i32 2, i32 %x
  %d = lshr i32 %c, 24
  ret i32 %d
}

; CHECK-LABEL: @four
define i32 @four(i32 %a) {
; CHECK: ret i32 0
  %x = add nsw i32 %a, 42
  %y = add nsw i32 %a, 64
  %b = icmp ugt i32 %y, 5
  %c = select i1 %b, i32 2, i32 %x
  %d = lshr i32 %c, 24
  ret i32 %d
}

; CHECK-LABEL: @four_swapped
define i32 @four_swapped(i32 %a) {
; CHECK: ret i32 %d
  %x = add nsw i32 %a, 42
  %y = add nsw i32 %a, 64
  %b = icmp ugt i32 %x, 5
  %c = select i1 %b, i32 2, i32 %y
  %d = lshr i32 %c, 24
  ret i32 %d
}