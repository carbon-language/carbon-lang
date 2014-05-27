; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck %s

define <32 x i8> @constant_pblendvb_avx2(<32 x i8> %xyzw, <32 x i8> %abcd) {
; CHECK-LABEL: constant_pblendvb_avx2:
; CHECK: vmovdqa
; CHECK: vpblendvb
  %1 = select <32 x i1> <i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false>, <32 x i8> %xyzw, <32 x i8> %abcd
  ret <32 x i8> %1
}

declare <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8>, <32 x i8>, <32 x i8>)
