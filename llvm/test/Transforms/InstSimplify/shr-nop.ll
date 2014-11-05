; RUN: opt < %s -instsimplify -S | FileCheck %s

; CHECK-LABEL: @foo
; CHECK:      %[[and:.*]] = and i32 %x, 1
; CHECK-NEXT: %[[add:.*]] = add i32 %[[and]], -1
; CHECK-NEXT: ret i32 %[[add]]
define i32 @foo(i32 %x) {
 %o = and i32 %x, 1
 %n = add i32 %o, -1
 %t = ashr i32 %n, 17
 ret i32 %t
}

; CHECK-LABEL: @exact_lshr_eq_both_zero
; CHECK-NEXT: ret i1 true
define i1 @exact_lshr_eq_both_zero(i8 %a) {
 %shr = lshr exact i8 0, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_eq_both_zero
; CHECK-NEXT: ret i1 true
define i1 @exact_ashr_eq_both_zero(i8 %a) {
 %shr = ashr exact i8 0, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_ashr_eq_both_zero
; CHECK-NEXT: ret i1 true
define i1 @nonexact_ashr_eq_both_zero(i8 %a) {
 %shr = ashr i8 0, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_ne_both_zero
; CHECK-NEXT: ret i1 false
define i1 @exact_lshr_ne_both_zero(i8 %a) {
 %shr = lshr exact i8 0, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_ne_both_zero
; CHECK-NEXT: ret i1 false
define i1 @exact_ashr_ne_both_zero(i8 %a) {
 %shr = ashr exact i8 0, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_lshr_ne_both_zero
; CHECK-NEXT: ret i1 false
define i1 @nonexact_lshr_ne_both_zero(i8 %a) {
 %shr = lshr i8 0, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_ashr_ne_both_zero
; CHECK-NEXT: ret i1 false
define i1 @nonexact_ashr_ne_both_zero(i8 %a) {
 %shr = ashr i8 0, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_eq_last_zero
; CHECK-NEXT: ret i1 false
define i1 @exact_lshr_eq_last_zero(i8 %a) {
 %shr = lshr exact i8 128, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_eq_last_zero
; CHECK-NEXT: ret i1 false
define i1 @exact_ashr_eq_last_zero(i8 %a) {
 %shr = ashr exact i8 -128, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_lshr_eq_both_zero
; CHECK-NEXT: ret i1 true
define i1 @nonexact_lshr_eq_both_zero(i8 %a) {
 %shr = lshr i8 0, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_ne_last_zero
; CHECK-NEXT: ret i1 true
define i1 @exact_lshr_ne_last_zero(i8 %a) {
 %shr = lshr exact i8 128, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_ne_last_zero
; CHECK-NEXT: ret i1 true
define i1 @exact_ashr_ne_last_zero(i8 %a) {
 %shr = ashr exact i8 -128, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_lshr_eq_last_zero
; CHECK-NEXT: ret i1 false
define i1 @nonexact_lshr_eq_last_zero(i8 %a) {
 %shr = lshr i8 128, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_ashr_eq_last_zero
; CHECK-NEXT: ret i1 false
define i1 @nonexact_ashr_eq_last_zero(i8 %a) {
 %shr = ashr i8 -128, %a
 %cmp = icmp eq i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_lshr_ne_last_zero
; CHECK-NEXT: ret i1 true
define i1 @nonexact_lshr_ne_last_zero(i8 %a) {
 %shr = lshr i8 128, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_ashr_ne_last_zero
; CHECK-NEXT: ret i1 true
define i1 @nonexact_ashr_ne_last_zero(i8 %a) {
 %shr = ashr i8 -128, %a
 %cmp = icmp ne i8 %shr, 0
 ret i1 %cmp
}

; CHECK-LABEL: @lshr_eq_first_zero
; CHECK-NEXT: ret i1 false
define i1 @lshr_eq_first_zero(i8 %a) {
 %shr = lshr i8 0, %a
 %cmp = icmp eq i8 %shr, 2
 ret i1 %cmp
}

; CHECK-LABEL: @ashr_eq_first_zero
; CHECK-NEXT: ret i1 false
define i1 @ashr_eq_first_zero(i8 %a) {
 %shr = ashr i8 0, %a
 %cmp = icmp eq i8 %shr, 2
 ret i1 %cmp
}

; CHECK-LABEL: @lshr_ne_first_zero
; CHECK-NEXT: ret i1 true
define i1 @lshr_ne_first_zero(i8 %a) {
 %shr = lshr i8 0, %a
 %cmp = icmp ne i8 %shr, 2
 ret i1 %cmp
}

; CHECK-LABEL: @ashr_ne_first_zero
; CHECK-NEXT: ret i1 true
define i1 @ashr_ne_first_zero(i8 %a) {
 %shr = ashr i8 0, %a
 %cmp = icmp ne i8 %shr, 2
 ret i1 %cmp
}

; CHECK-LABEL: @ashr_eq_both_minus1
; CHECK-NEXT: ret i1 true
define i1 @ashr_eq_both_minus1(i8 %a) {
 %shr = ashr i8 -1, %a
 %cmp = icmp eq i8 %shr, -1
 ret i1 %cmp
}

; CHECK-LABEL: @ashr_ne_both_minus1
; CHECK-NEXT: ret i1 false
define i1 @ashr_ne_both_minus1(i8 %a) {
 %shr = ashr i8 -1, %a
 %cmp = icmp ne i8 %shr, -1
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_eq_both_minus1
; CHECK-NEXT: ret i1 true
define i1 @exact_ashr_eq_both_minus1(i8 %a) {
 %shr = ashr exact i8 -1, %a
 %cmp = icmp eq i8 %shr, -1
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_ne_both_minus1
; CHECK-NEXT: ret i1 false
define i1 @exact_ashr_ne_both_minus1(i8 %a) {
 %shr = ashr exact i8 -1, %a
 %cmp = icmp ne i8 %shr, -1
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_eq_opposite_msb
; CHECK-NEXT: ret i1 false
define i1 @exact_ashr_eq_opposite_msb(i8 %a) {
 %shr = ashr exact i8 -128, %a
 %cmp = icmp eq i8 %shr, 1
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_eq_noexactlog
; CHECK-NEXT: ret i1 false
define i1 @exact_ashr_eq_noexactlog(i8 %a) {
 %shr = ashr exact i8 -90, %a
 %cmp = icmp eq i8 %shr, -30
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_ne_opposite_msb
; CHECK-NEXT: ret i1 true
define i1 @exact_ashr_ne_opposite_msb(i8 %a) {
 %shr = ashr exact i8 -128, %a
 %cmp = icmp ne i8 %shr, 1
 ret i1 %cmp
}

; CHECK-LABEL: @ashr_eq_opposite_msb
; CHECK-NEXT: ret i1 false
define i1 @ashr_eq_opposite_msb(i8 %a) {
 %shr = ashr i8 -128, %a
 %cmp = icmp eq i8 %shr, 1
 ret i1 %cmp
}

; CHECK-LABEL: @ashr_ne_opposite_msb
; CHECK-NEXT: ret i1 true
define i1 @ashr_ne_opposite_msb(i8 %a) {
 %shr = ashr i8 -128, %a
 %cmp = icmp ne i8 %shr, 1
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_eq_shift_gt
; CHECK-NEXT : ret i1 false
define i1 @exact_ashr_eq_shift_gt(i8 %a) {
 %shr = ashr exact i8 -2, %a
 %cmp = icmp eq i8 %shr, -8
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_ne_shift_gt
; CHECK-NEXT : ret i1 true
define i1 @exact_ashr_ne_shift_gt(i8 %a) {
 %shr = ashr exact i8 -2, %a
 %cmp = icmp ne i8 %shr, -8
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_ashr_eq_shift_gt
; CHECK-NEXT : ret i1 false
define i1 @nonexact_ashr_eq_shift_gt(i8 %a) {
 %shr = ashr i8 -2, %a
 %cmp = icmp eq i8 %shr, -8
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_ashr_ne_shift_gt
; CHECK-NEXT : ret i1 true
define i1 @nonexact_ashr_ne_shift_gt(i8 %a) {
 %shr = ashr i8 -2, %a
 %cmp = icmp ne i8 %shr, -8
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_eq_shift_gt
; CHECK-NEXT: ret i1 false
define i1 @exact_lshr_eq_shift_gt(i8 %a) {
 %shr = lshr exact i8 2, %a
 %cmp = icmp eq i8 %shr, 8
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_ne_shift_gt
; CHECK-NEXT: ret i1 true
define i1 @exact_lshr_ne_shift_gt(i8 %a) {
 %shr = lshr exact i8 2, %a
 %cmp = icmp ne i8 %shr, 8
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_lshr_eq_shift_gt
; CHECK-NEXT : ret i1 false
define i1 @nonexact_lshr_eq_shift_gt(i8 %a) {
 %shr = lshr i8 2, %a
 %cmp = icmp eq i8 %shr, 8
 ret i1 %cmp
}

; CHECK-LABEL: @nonexact_lshr_ne_shift_gt
; CHECK-NEXT : ret i1 true
define i1 @nonexact_lshr_ne_shift_gt(i8 %a) {
 %shr = ashr i8 2, %a
 %cmp = icmp ne i8 %shr, 8
 ret i1 %cmp
}

; CHECK-LABEL: @exact_ashr_ne_noexactlog
; CHECK-NEXT: ret i1 true
define i1 @exact_ashr_ne_noexactlog(i8 %a) {
 %shr = ashr exact i8 -90, %a
 %cmp = icmp ne i8 %shr, -30
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_eq_noexactlog
; CHECK-NEXT: ret i1 false
define i1 @exact_lshr_eq_noexactlog(i8 %a) {
 %shr = lshr exact i8 90, %a
 %cmp = icmp eq i8 %shr, 30
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_ne_noexactlog
; CHECK-NEXT: ret i1 true
define i1 @exact_lshr_ne_noexactlog(i8 %a) {
 %shr = lshr exact i8 90, %a
 %cmp = icmp ne i8 %shr, 30
 ret i1 %cmp
}

; CHECK-LABEL: @exact_lshr_lowbit
; CHECK-NEXT: ret i32 7
define i32 @exact_lshr_lowbit(i32 %shiftval) {
  %shr = lshr exact i32 7, %shiftval
  ret i32 %shr
}

; CHECK-LABEL: @exact_ashr_lowbit
; CHECK-NEXT: ret i32 7
define i32 @exact_ashr_lowbit(i32 %shiftval) {
  %shr = ashr exact i32 7, %shiftval
  ret i32 %shr
}
