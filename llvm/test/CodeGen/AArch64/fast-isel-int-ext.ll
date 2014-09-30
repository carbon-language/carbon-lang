; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort -verify-machineinstrs < %s | FileCheck %s

;
; Test that we only use the sign/zero extend in the address calculation when
; necessary.
;
; SHIFT
;
define i64 @load_addr_shift_zext1(i32 zeroext %a, i64 %b) {
; CHECK-LABEL: load_addr_shift_zext1
; CHECK:       ldr {{x[0-9]+}}, [x1, x0, lsl #3]
  %1 = zext i32 %a to i64
  %2 = shl i64 %1, 3
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

define i64 @load_addr_shift_zext2(i32 signext %a, i64 %b) {
; CHECK-LABEL: load_addr_shift_zext2
; CHECK:       ldr {{x[0-9]+}}, [x1, w0, uxtw #3{{\]}}
  %1 = zext i32 %a to i64
  %2 = shl i64 %1, 3
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

define i64 @load_addr_shift_sext1(i32 signext %a, i64 %b) {
; CHECK-LABEL: load_addr_shift_sext1
; CHECK:       ldr {{x[0-9]+}}, [x1, x0, lsl #3]
  %1 = sext i32 %a to i64
  %2 = shl i64 %1, 3
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

define i64 @load_addr_shift_sext2(i32 zeroext %a, i64 %b) {
; CHECK-LABEL: load_addr_shift_sext2
; CHECK:       ldr {{x[0-9]+}}, [x1, w0, sxtw #3]
  %1 = sext i32 %a to i64
  %2 = shl i64 %1, 3
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

;
; MUL
;
define i64 @load_addr_mul_zext1(i32 zeroext %a, i64 %b) {
; CHECK-LABEL: load_addr_mul_zext1
; CHECK:       ldr {{x[0-9]+}}, [x1, x0, lsl #3]
  %1 = zext i32 %a to i64
  %2 = mul i64 %1, 8
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

define i64 @load_addr_mul_zext2(i32 signext %a, i64 %b) {
; CHECK-LABEL: load_addr_mul_zext2
; CHECK:       ldr {{x[0-9]+}}, [x1, w0, uxtw #3]
  %1 = zext i32 %a to i64
  %2 = mul i64 %1, 8
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

define i64 @load_addr_mul_sext1(i32 signext %a, i64 %b) {
; CHECK-LABEL: load_addr_mul_sext1
; CHECK:       ldr {{x[0-9]+}}, [x1, x0, lsl #3]
  %1 = sext i32 %a to i64
  %2 = mul i64 %1, 8
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

define i64 @load_addr_mul_sext2(i32 zeroext %a, i64 %b) {
; CHECK-LABEL: load_addr_mul_sext2
; CHECK:       ldr {{x[0-9]+}}, [x1, w0, sxtw #3]
  %1 = sext i32 %a to i64
  %2 = mul i64 %1, 8
  %3 = add i64 %b, %2
  %4 = inttoptr i64 %3 to i64*
  %5 = load i64* %4
  ret i64 %5
}

; Test folding of the sign-/zero-extend into the load instruction.
define i32 @load_zext_i8_to_i32(i8* %a) {
; CHECK-LABEL: load_zext_i8_to_i32
; CHECK:       ldrb w0, [x0]
; CHECK-NOT:   uxtb
  %1 = load i8* %a
  %2 = zext i8 %1 to i32
  ret i32 %2
}

define i32 @load_zext_i16_to_i32(i16* %a) {
; CHECK-LABEL: load_zext_i16_to_i32
; CHECK:       ldrh w0, [x0]
; CHECK-NOT:   uxth
  %1 = load i16* %a
  %2 = zext i16 %1 to i32
  ret i32 %2
}

define i64 @load_zext_i8_to_i64(i8* %a) {
; CHECK-LABEL: load_zext_i8_to_i64
; CHECK:       ldrb w0, [x0]
; CHECK-NOT:   uxtb
  %1 = load i8* %a
  %2 = zext i8 %1 to i64
  ret i64 %2
}

define i64 @load_zext_i16_to_i64(i16* %a) {
; CHECK-LABEL: load_zext_i16_to_i64
; CHECK:       ldrh w0, [x0]
; CHECK-NOT:   uxth
  %1 = load i16* %a
  %2 = zext i16 %1 to i64
  ret i64 %2
}

define i64 @load_zext_i32_to_i64(i32* %a) {
; CHECK-LABEL: load_zext_i32_to_i64
; CHECK:       ldr w0, [x0]
; CHECK-NOT:   uxtw
  %1 = load i32* %a
  %2 = zext i32 %1 to i64
  ret i64 %2
}

define i32 @load_sext_i8_to_i32(i8* %a) {
; CHECK-LABEL: load_sext_i8_to_i32
; CHECK:       ldrsb w0, [x0]
; CHECK-NOT:   sxtb
  %1 = load i8* %a
  %2 = sext i8 %1 to i32
  ret i32 %2
}

define i32 @load_sext_i16_to_i32(i16* %a) {
; CHECK-LABEL: load_sext_i16_to_i32
; CHECK:       ldrsh w0, [x0]
; CHECK-NOT:   sxth
  %1 = load i16* %a
  %2 = sext i16 %1 to i32
  ret i32 %2
}

define i64 @load_sext_i8_to_i64(i8* %a) {
; CHECK-LABEL: load_sext_i8_to_i64
; CHECK:       ldrsb w0, [x0]
; CHECK-NOT:   sxtb
  %1 = load i8* %a
  %2 = sext i8 %1 to i64
  ret i64 %2
}

define i64 @load_sext_i16_to_i64(i16* %a) {
; CHECK-LABEL: load_sext_i16_to_i64
; CHECK:       ldrsh w0, [x0]
; CHECK-NOT:   sxth
  %1 = load i16* %a
  %2 = sext i16 %1 to i64
  ret i64 %2
}

define i64 @load_sext_i32_to_i64(i32* %a) {
; CHECK-LABEL: load_sext_i32_to_i64
; CHECK:       ldrsw x0, [x0]
; CHECK-NOT:   sxtw
  %1 = load i32* %a
  %2 = sext i32 %1 to i64
  ret i64 %2
}

