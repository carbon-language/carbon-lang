; RUN: opt -S -instcombine < %s | FileCheck %s

define i32 @compute_min_2(i32 %x, i32 %y) {
; CHECK-LABEL: compute_min_2
 entry:
  %not_x = sub i32 -1, %x
  %not_y = sub i32 -1, %y
  %cmp = icmp sgt i32 %not_x, %not_y
  %not_min = select i1 %cmp, i32 %not_x, i32 %not_y
  %min = sub i32 -1, %not_min
  ret i32 %min

; CHECK: %0 = icmp slt i32 %x, %y
; CHECK-NEXT: %1 = select i1 %0, i32 %x, i32 %y
; CHECK-NEXT: ret i32 %1
}

define i32 @compute_min_3(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: compute_min_3
 entry:
  %not_x = sub i32 -1, %x
  %not_y = sub i32 -1, %y
  %not_z = sub i32 -1, %z
  %cmp_1 = icmp sgt i32 %not_x, %not_y
  %not_min_1 = select i1 %cmp_1, i32 %not_x, i32 %not_y
  %cmp_2 = icmp sgt i32 %not_min_1, %not_z
  %not_min_2 = select i1 %cmp_2, i32 %not_min_1, i32 %not_z
  %min = sub i32 -1, %not_min_2
  ret i32 %min

; CHECK: %0 = icmp slt i32 %x, %y
; CHECK-NEXT: %1 = select i1 %0, i32 %x, i32 %y
; CHECK-NEXT: %2 = icmp slt i32 %1, %z
; CHECK-NEXT: %3 = select i1 %2, i32 %1, i32 %z
; CHECK-NEXT: ret i32 %3
}

define i32 @compute_min_arithmetic(i32 %x, i32 %y) {
; CHECK-LABEL: compute_min_arithmetic
 entry:
  %not_value = sub i32 3, %x
  %not_y = sub i32 -1, %y
  %cmp = icmp sgt i32 %not_value, %not_y
  %not_min = select i1 %cmp, i32 %not_value, i32 %not_y
  ret i32 %not_min

; CHECK: %0 = add i32 %x, -4
; CHECK-NEXT: %1 = icmp slt i32 %0, %y
; CHECK-NEXT: %2 = select i1 %1, i32 %0, i32 %y
; CHECK-NEXT: %3 = xor i32 %2, -1
; CHECK-NEXT: ret i32 %3
}

declare void @fake_use(i32)

define i32 @compute_min_pessimization(i32 %x, i32 %y) {
; CHECK-LABEL: compute_min_pessimization
 entry:
  %not_value = sub i32 3, %x
  call void @fake_use(i32 %not_value)
  %not_y = sub i32 -1, %y
  %cmp = icmp sgt i32 %not_value, %not_y
; CHECK: %not_value = sub i32 3, %x
; CHECK: %cmp = icmp sgt i32 %not_value, %not_y
  %not_min = select i1 %cmp, i32 %not_value, i32 %not_y
  %min = sub i32 -1, %not_min
  ret i32 %min
}

define i32 @max_of_nots(i32 %x, i32 %y) {
; CHECK-LABEL: @max_of_nots(
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: xor
; CHECK-NEXT: ret
  %c0 = icmp sgt i32 %y, 0
  %xor_y = xor i32 %y, -1
  %s0 = select i1 %c0, i32 %xor_y, i32 -1
  %xor_x = xor i32 %x, -1
  %c1 = icmp slt i32 %s0, %xor_x
  %smax96 = select i1 %c1, i32 %xor_x, i32 %s0
  ret i32 %smax96
}
