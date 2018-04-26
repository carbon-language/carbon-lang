; RUN: opt < %s -instcombine -S | FileCheck %s
; ModuleID = 'test/Transforms/InstCombine/add4.ll'
source_filename = "test/Transforms/InstCombine/add4.ll"

define i64 @match_unsigned(i64 %x) {
; CHECK-LABEL: @match_unsigned(
; CHECK:    [[TMP:%.*]] = add
; CHECK-NEXT:    ret i64 [[TMP]]
;
bb:
  %tmp = urem i64 %x, 299
  %tmp1 = udiv i64 %x, 299
  %tmp2 = urem i64 %tmp1, 64
  %tmp3 = mul i64 %tmp2, 299
  %tmp4 = add nuw nsw i64 %tmp, %tmp3
  ret i64 %tmp4
}

define i64 @match_andAsRem_lshrAsDiv_shlAsMul(i64 %x) {
; CHECK-LABEL: @match_andAsRem_lshrAsDiv_shlAsMul(
; CHECK:    [[TMP:%.*]] = or
; CHECK-NEXT:    ret i64 [[TMP]]
;
bb:
  %tmp = and i64 %x, 63
  %tmp1 = lshr i64 %x, 6
  %tmp2 = urem i64 %tmp1, 9
  %tmp3 = shl nuw nsw i64 %tmp2, 6
  %tmp4 = add nuw nsw i64 %tmp, %tmp3
  ret i64 %tmp4
}

define i64 @match_signed(i64 %x) {
; CHECK-LABEL: @match_signed(
; CHECK:    [[TMP1:%.*]] = add
; CHECK:    [[TMP2:%.*]] = add
; CHECK-NEXT:    ret i64 [[TMP2]]
;
bb:
  %tmp = srem i64 %x, 299
  %tmp1 = sdiv i64 %x, 299
  %tmp2 = srem i64 %tmp1, 64
  %tmp3 = sdiv i64 %x, 19136
  %tmp4 = srem i64 %tmp3, 9
  %tmp5 = mul nuw nsw i64 %tmp2, 299
  %tmp6 = add nuw nsw i64 %tmp, %tmp5
  %tmp7 = mul nuw nsw i64 %tmp4, 19136
  %tmp8 = add nuw nsw i64 %tmp6, %tmp7
  ret i64 %tmp8
}

define i64 @not_match_inconsistent_signs(i64 %x) {
; CHECK-LABEL: @not_match_inconsistent_signs(
; CHECK:    [[TMP:%.*]] = add
; CHECK-NEXT:    ret i64 [[TMP]]
;
bb:
  %tmp = urem i64 %x, 299
  %tmp1 = sdiv i64 %x, 299
  %tmp2 = urem i64 %tmp1, 64
  %tmp3 = mul i64 %tmp2, 299
  %tmp4 = add nuw nsw i64 %tmp, %tmp3
  ret i64 %tmp4
}

define i64 @not_match_inconsistent_values(i64 %x) {
; CHECK-LABEL: @not_match_inconsistent_values(
; CHECK:    [[TMP:%.*]] = add
; CHECK-NEXT:    ret i64 [[TMP]]
;
bb:
  %tmp = urem i64 %x, 299
  %tmp1 = udiv i64 %x, 29
  %tmp2 = urem i64 %tmp1, 64
  %tmp3 = mul i64 %tmp2, 299
  %tmp4 = add nuw nsw i64 %tmp, %tmp3
  ret i64 %tmp4
}
