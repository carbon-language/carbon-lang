; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; This test verifies that we can enable subtarget features via
; the function attributes and generate appropriate code (or,
; in this case, select the instruction at all).

; Function Attrs: nounwind
define <16 x i8> @foo(<16 x i8> %data, <16 x i8> %key) #0 {
entry:
  %__p0.addr.i = alloca <16 x i8>, align 16
  %__p1.addr.i = alloca <16 x i8>, align 16
  %__ret.i = alloca <16 x i8>, align 16
  %data.addr = alloca <16 x i8>, align 16
  %key.addr = alloca <16 x i8>, align 16
  store <16 x i8> %data, <16 x i8>* %data.addr, align 16
  store <16 x i8> %key, <16 x i8>* %key.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %data.addr, align 16
  %1 = load <16 x i8>, <16 x i8>* %key.addr, align 16
  store <16 x i8> %0, <16 x i8>* %__p0.addr.i, align 16
  store <16 x i8> %1, <16 x i8>* %__p1.addr.i, align 16
  %2 = load <16 x i8>, <16 x i8>* %__p0.addr.i, align 16
  %3 = load <16 x i8>, <16 x i8>* %__p1.addr.i, align 16
  %vaeseq_v.i = call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %2, <16 x i8> %3)
  store <16 x i8> %vaeseq_v.i, <16 x i8>* %__ret.i, align 16
  %4 = load <16 x i8>, <16 x i8>* %__ret.i, align 16
  ret <16 x i8> %4
}

; CHECK: foo
; CHECK: aese

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8>, <16 x i8>)

attributes #0 = { nounwind "target-features"="+neon,+crc,+crypto" }
