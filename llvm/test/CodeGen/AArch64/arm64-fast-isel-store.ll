; RUN: llc -mtriple=aarch64-unknown-unknown                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-unknown-unknown -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define void @store_i8(i8* %a) {
; CHECK-LABEL: store_i8
; CHECK: strb  wzr, [x0]
  store i8 0, i8* %a
  ret void
}

define void @store_i16(i16* %a) {
; CHECK-LABEL: store_i16
; CHECK: strh  wzr, [x0]
  store i16 0, i16* %a
  ret void
}

define void @store_i32(i32* %a) {
; CHECK-LABEL: store_i32
; CHECK: str  wzr, [x0]
  store i32 0, i32* %a
  ret void
}

define void @store_i64(i64* %a) {
; CHECK-LABEL: store_i64
; CHECK: str  xzr, [x0]
  store i64 0, i64* %a
  ret void
}
