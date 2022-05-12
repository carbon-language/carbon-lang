; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -verify-machineinstrs

define void @f() {
  %a = alloca i8, i32 8, align 8
  %b = alloca i8, i32 8, align 8

  %c = bitcast i8* %a to i64*
  %d = bitcast i8* %b to i64*

  store atomic i64 0, i64* %c seq_cst, align 8
  store atomic i64 0, i64* %d seq_cst, align 8

  %e = load atomic i64, i64* %d seq_cst, align 8

  ret void
}
