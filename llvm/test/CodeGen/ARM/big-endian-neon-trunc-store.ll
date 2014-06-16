; RUN: llc < %s -mtriple armeb-eabi -mattr v7,neon -o - | FileCheck %s

define void @vector_trunc_store_2i64_to_2i16( <2 x i64>* %loadaddr, <2 x i16>* %storeaddr ) {
; CHECK-LABEL: vector_trunc_store_2i64_to_2i16:
; CHECK:       vmovn.i64  [[REG:d[0-9]+]]
; CHECK:       vrev32.16  [[REG]], [[REG]]
; CHECK:       vuzp.16    [[REG]], [[REG2:d[0-9]+]]
; CHECK:       vrev32.16  [[REG]], [[REG2]]
  %1 = load <2 x i64>* %loadaddr
  %2 = trunc <2 x i64> %1 to <2 x i16>
  store <2 x i16> %2, <2 x i16>* %storeaddr
  ret void
}

define void @vector_trunc_store_4i32_to_4i8( <4 x i32>* %loadaddr, <4 x i8>* %storeaddr ) {
; CHECK-LABEL: vector_trunc_store_4i32_to_4i8:
; CHECK:       vmovn.i32 [[REG:d[0-9]+]]
; CHECK:       vrev16.8  [[REG]], [[REG]]
; CHECK:       vuzp.8    [[REG]], [[REG2:d[0-9]+]]
; CHECK:       vrev32.8  [[REG]], [[REG2]]
  %1 = load <4 x i32>* %loadaddr
  %2 = trunc <4 x i32> %1 to <4 x i8>
  store <4 x i8> %2, <4 x i8>* %storeaddr
  ret void
}

