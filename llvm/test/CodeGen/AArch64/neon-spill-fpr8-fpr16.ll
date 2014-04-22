; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; not relevant for arm64: <1 x iN> isn't legal

; This file tests the spill of FPR8/FPR16. The volatile loads/stores force the
; allocator to keep the value live until it's needed.

%bigtype_v1i8 = type [20 x <1 x i8>]

define void @spill_fpr8(%bigtype_v1i8* %addr) {
; CHECK-LABEL: spill_fpr8:
; CHECK: 1-byte Folded Spill
; CHECK: 1-byte Folded Reload
  %val1 = load volatile %bigtype_v1i8* %addr
  %val2 = load volatile %bigtype_v1i8* %addr
  store volatile %bigtype_v1i8 %val1, %bigtype_v1i8* %addr
  store volatile %bigtype_v1i8 %val2, %bigtype_v1i8* %addr
  ret void
}

%bigtype_v1i16 = type [20 x <1 x i16>]

define void @spill_fpr16(%bigtype_v1i16* %addr) {
; CHECK-LABEL: spill_fpr16:
; CHECK: 2-byte Folded Spill
; CHECK: 2-byte Folded Reload
  %val1 = load volatile %bigtype_v1i16* %addr
  %val2 = load volatile %bigtype_v1i16* %addr
  store volatile %bigtype_v1i16 %val1, %bigtype_v1i16* %addr
  store volatile %bigtype_v1i16 %val2, %bigtype_v1i16* %addr
  ret void
}
