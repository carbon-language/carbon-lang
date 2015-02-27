; RUN: llc -march=mipsel -O3 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-none-nacl-gnu -O3 < %s \
; RUN:  | FileCheck %s -check-prefix=CHECK-NACL

@var = external global i32

define void @f() {
  %val1 = load volatile i32, i32* @var
  %val2 = load volatile i32, i32* @var
  %val3 = load volatile i32, i32* @var
  %val4 = load volatile i32, i32* @var
  %val5 = load volatile i32, i32* @var
  %val6 = load volatile i32, i32* @var
  %val7 = load volatile i32, i32* @var
  %val8 = load volatile i32, i32* @var
  %val9 = load volatile i32, i32* @var
  %val10 = load volatile i32, i32* @var
  %val11 = load volatile i32, i32* @var
  %val12 = load volatile i32, i32* @var
  %val13 = load volatile i32, i32* @var
  %val14 = load volatile i32, i32* @var
  %val15 = load volatile i32, i32* @var
  %val16 = load volatile i32, i32* @var
  store volatile i32 %val1, i32* @var
  store volatile i32 %val2, i32* @var
  store volatile i32 %val3, i32* @var
  store volatile i32 %val4, i32* @var
  store volatile i32 %val5, i32* @var
  store volatile i32 %val6, i32* @var
  store volatile i32 %val7, i32* @var
  store volatile i32 %val8, i32* @var
  store volatile i32 %val9, i32* @var
  store volatile i32 %val10, i32* @var
  store volatile i32 %val11, i32* @var
  store volatile i32 %val12, i32* @var
  store volatile i32 %val13, i32* @var
  store volatile i32 %val14, i32* @var
  store volatile i32 %val15, i32* @var
  store volatile i32 %val16, i32* @var
  ret void

; Check that t6, t7 and t8 are used in non-NaCl code.
; CHECK:    lw  $14
; CHECK:    lw  $15
; CHECK:    lw  $24

; t6, t7 and t8 are reserved in NaCl.
; CHECK-NACL-NOT:    lw  $14
; CHECK-NACL-NOT:    lw  $15
; CHECK-NACL-NOT:    lw  $24
}
