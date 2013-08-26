; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

; zext

define i32 @zext_8_32(i8 %a) nounwind ssp {
; ELF64: zext_8_32
  %r = zext i8 %a to i32
; ELF64: rlwinm {{[0-9]+}}, {{[0-9]+}}, 0, 24, 31
  ret i32 %r
}

define i32 @zext_16_32(i16 %a) nounwind ssp {
; ELF64: zext_16_32
  %r = zext i16 %a to i32
; ELF64: rlwinm {{[0-9]+}}, {{[0-9]+}}, 0, 16, 31
  ret i32 %r
}

define i64 @zext_8_64(i8 %a) nounwind ssp {
; ELF64: zext_8_64
  %r = zext i8 %a to i64
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 56
  ret i64 %r
}

define i64 @zext_16_64(i16 %a) nounwind ssp {
; ELF64: zext_16_64
  %r = zext i16 %a to i64
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 48
  ret i64 %r
}

define i64 @zext_32_64(i32 %a) nounwind ssp {
; ELF64: zext_32_64
  %r = zext i32 %a to i64
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 32
  ret i64 %r
}

; sext

define i32 @sext_8_32(i8 %a) nounwind ssp {
; ELF64: sext_8_32
  %r = sext i8 %a to i32
; ELF64: extsb
  ret i32 %r
}

define i32 @sext_16_32(i16 %a) nounwind ssp {
; ELF64: sext_16_32
  %r = sext i16 %a to i32
; ELF64: extsh
  ret i32 %r
}

define i64 @sext_8_64(i8 %a) nounwind ssp {
; ELF64: sext_8_64
  %r = sext i8 %a to i64
; ELF64: extsb
  ret i64 %r
}

define i64 @sext_16_64(i16 %a) nounwind ssp {
; ELF64: sext_16_64
  %r = sext i16 %a to i64
; ELF64: extsh
  ret i64 %r
}

define i64 @sext_32_64(i32 %a) nounwind ssp {
; ELF64: sext_32_64
  %r = sext i32 %a to i64
; ELF64: extsw
  ret i64 %r
}
