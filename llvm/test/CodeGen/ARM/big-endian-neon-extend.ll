; RUN: llc < %s -mtriple armeb-eabi -mattr v7,neon -o - | FileCheck %s

define void @vector_ext_2i8_to_2i64( <2 x i8>* %loadaddr, <2 x i64>* %storeaddr ) {
; CHECK-LABEL: vector_ext_2i8_to_2i64:
; CHECK:       vld1.16 {[[REG:d[0-9]+]]
; CHECK:       vmov.i64 {{q[0-9]+}}, #0xff
; CHECK:       vrev16.8  [[REG]], [[REG]]
; CHECK:       vmovl.u8  {{q[0-9]+}}, [[REG]]
  %1 = load <2 x i8>* %loadaddr
  %2 = zext <2 x i8> %1 to <2 x i64>
  store <2 x i64> %2, <2 x i64>* %storeaddr
  ret void
}

define void @vector_ext_2i16_to_2i64( <2 x i16>* %loadaddr, <2 x i64>* %storeaddr ) {
; CHECK-LABEL: vector_ext_2i16_to_2i64:
; CHECK:       vld1.32 {[[REG:d[0-9]+]]
; CHECK:       vmov.i64 {{q[0-9]+}}, #0xffff
; CHECK:       vrev32.16  [[REG]], [[REG]]
; CHECK:       vmovl.u16  {{q[0-9]+}}, [[REG]]
  %1 = load <2 x i16>* %loadaddr
  %2 = zext <2 x i16> %1 to <2 x i64>
  store <2 x i64> %2, <2 x i64>* %storeaddr
  ret void
}


define void @vector_ext_2i8_to_2i32( <2 x i8>* %loadaddr, <2 x i32>* %storeaddr ) {
; CHECK-LABEL: vector_ext_2i8_to_2i32:
; CHECK:       vld1.16 {[[REG:d[0-9]+]]
; CHECK:       vrev16.8  [[REG]], [[REG]]
  %1 = load <2 x i8>* %loadaddr
  %2 = zext <2 x i8> %1 to <2 x i32>
  store <2 x i32> %2, <2 x i32>* %storeaddr
  ret void
}

define void @vector_ext_2i16_to_2i32( <2 x i16>* %loadaddr, <2 x i32>* %storeaddr ) {
; CHECK-LABEL: vector_ext_2i16_to_2i32:
; CHECK:       vld1.32 {[[REG:d[0-9]+]]
; CHECK:       vrev32.16  [[REG]], [[REG]]
; CHECK:       vmovl.u16  {{q[0-9]+}}, [[REG]]
  %1 = load <2 x i16>* %loadaddr
  %2 = zext <2 x i16> %1 to <2 x i32>
  store <2 x i32> %2, <2 x i32>* %storeaddr
  ret void
}

define void @vector_ext_2i8_to_2i16( <2 x i8>* %loadaddr, <2 x i16>* %storeaddr ) {
; CHECK-LABEL: vector_ext_2i8_to_2i16:
; CHECK:       vld1.16 {[[REG:d[0-9]+]]
; CHECK:       vrev16.8  [[REG]], [[REG]]
; CHECK:       vmovl.u8  {{q[0-9]+}}, [[REG]]
  %1 = load <2 x i8>* %loadaddr
  %2 = zext <2 x i8> %1 to <2 x i16>
  store <2 x i16> %2, <2 x i16>* %storeaddr
  ret void
}

define void @vector_ext_4i8_to_4i32( <4 x i8>* %loadaddr, <4 x i32>* %storeaddr ) {
; CHECK-LABEL: vector_ext_4i8_to_4i32:
; CHECK:       vld1.32 {[[REG:d[0-9]+]]
; CHECK:       vrev32.8  [[REG]], [[REG]]
; CHECK:       vmovl.u8  {{q[0-9]+}}, [[REG]]
  %1 = load <4 x i8>* %loadaddr
  %2 = zext <4 x i8> %1 to <4 x i32>
  store <4 x i32> %2, <4 x i32>* %storeaddr
  ret void
}

define void @vector_ext_4i8_to_4i16( <4 x i8>* %loadaddr, <4 x i16>* %storeaddr ) {
; CHECK-LABEL: vector_ext_4i8_to_4i16:
; CHECK:       vld1.32 {[[REG:d[0-9]+]]
; CHECK:       vrev32.8  [[REG]], [[REG]]
; CHECK:       vmovl.u8  {{q[0-9]+}}, [[REG]]
  %1 = load <4 x i8>* %loadaddr
  %2 = zext <4 x i8> %1 to <4 x i16>
  store <4 x i16> %2, <4 x i16>* %storeaddr
  ret void
}

