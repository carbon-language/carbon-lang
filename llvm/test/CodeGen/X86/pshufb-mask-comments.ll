; RUN: llc < %s -march=x86-64 -mattr=+ssse3 | FileCheck %s

; Test that the pshufb mask comment is correct.

define <16 x i8> @test1(<16 x i8> %V) {
; CHECK-LABEL: test1:
; CHECK: pshufb {{.*}}# xmm0 = xmm0[1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4]
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %V, <16 x i8> <i8 1, i8 0, i8 0, i8 0, i8 0, i8 2, i8 0, i8 0, i8 0, i8 0, i8 3, i8 0, i8 0, i8 0, i8 0, i8 4>)
  ret <16 x i8> %1
}

; Test that indexes larger than the size of the vector are shown masked (bottom 4 bits).

define <16 x i8> @test2(<16 x i8> %V) {
; CHECK-LABEL: test2:
; CHECK: pshufb {{.*}}# xmm0 = xmm0[15,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2]
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %V, <16 x i8> <i8 15, i8 0, i8 0, i8 0, i8 0, i8 16, i8 0, i8 0, i8 0, i8 0, i8 17, i8 0, i8 0, i8 0, i8 0, i8 50>)
  ret <16 x i8> %1
}

; Test that indexes with bit seven set are shown as zero.

define <16 x i8> @test3(<16 x i8> %V) {
; CHECK-LABEL: test3:
; CHECK: pshufb {{.*}}# xmm0 = xmm0[1,0,0,15,0,2,0,0],zero,xmm0[0,3,0,0],zero,xmm0[0,4]
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %V, <16 x i8> <i8 1, i8 0, i8 0, i8 127, i8 0, i8 2, i8 0, i8 0, i8 128, i8 0, i8 3, i8 0, i8 0, i8 255, i8 0, i8 4>)
  ret <16 x i8> %1
}

declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>) nounwind readnone
