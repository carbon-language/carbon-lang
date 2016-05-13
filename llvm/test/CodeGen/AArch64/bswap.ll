; RUN: llc -march=aarch64 < %s | FileCheck %s

; Test byte swap instrinsic lowering on AArch64 targets.

define i16 @bswap16(i16 %x) #0 {
  %1 = tail call i16 @llvm.bswap.i16(i16 %x)
  ret i16 %1
; CHECK-LABEL: bswap16
; CHECK:       rev [[R0:w[0-9]+]], {{w[0-9]+}}
; CHECK-NEXT:  lsr {{w[0-9]+}}, [[R0]], #16
}

define i32 @bswap32(i32 %x) #0 {
  %1 = tail call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %1
; CHECK-LABEL: bswap32
; CHECK:       rev [[R0:w[0-9]+]], [[R0]]
}

define i48 @bswap48(i48 %x) #0 {
  %1 = tail call i48 @llvm.bswap.i48(i48 %x)
  ret i48 %1
; CHECK-LABEL: bswap48
; CHECK:       rev [[R0:x[0-9]+]], {{x[0-9]+}}
; CHECK-NEXT:  lsr {{x[0-9]+}}, [[R0]], #16
}

define i64 @bswap64(i64 %x) #0 {
  %1 = tail call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %1
; CHECK-LABEL: bswap64
; CHECK:       rev [[R0:x[0-9]+]], [[R0]]
; CHECK-NOT:   rev
}

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
declare i48 @llvm.bswap.i48(i48)
declare i64 @llvm.bswap.i64(i64)
