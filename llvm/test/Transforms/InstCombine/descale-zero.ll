; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define internal i8* @descale_zero() {
entry:
; CHECK: load i16** inttoptr (i64 48 to i16**), align 16
; CHECK-NEXT: bitcast i16*
; CHECK-NEXT: ret i8*
  %i16_ptr = load i16** inttoptr (i64 48 to i16**), align 16
  %num = load i64* inttoptr (i64 64 to i64*), align 64
  %num_times_2 = shl i64 %num, 1
  %num_times_2_plus_4 = add i64 %num_times_2, 4
  %i8_ptr = bitcast i16* %i16_ptr to i8*
  %i8_ptr_num_times_2_plus_4 = getelementptr i8* %i8_ptr, i64 %num_times_2_plus_4
  %num_times_neg2 = mul i64 %num, -2
  %num_times_neg2_minus_4 = add i64 %num_times_neg2, -4
  %addr = getelementptr i8* %i8_ptr_num_times_2_plus_4, i64 %num_times_neg2_minus_4
  ret i8* %addr
}
