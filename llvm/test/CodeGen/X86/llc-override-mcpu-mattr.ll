; RUN: llc < %s -march x86-64 -mcpu=broadwell | FileCheck %s
; RUN: llc < %s -march x86-64 -mattr=+avx2 | FileCheck %s

; Check that llc can overide function attributes target-cpu and target-features
; using command line options -mcpu and -mattr.

; CHECK: vpsadbw %ymm{{[0-9]+}}, %ymm{{[0-9]+}}, %ymm{{[0-9]+}}

define <4 x i64> @foo1(<4 x i64> %s1, <4 x i64> %s2) #0 {
entry:
  %0 = bitcast <4 x i64> %s1 to <32 x i8>
  %1 = bitcast <4 x i64> %s2 to <32 x i8>
  %2 = tail call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %0, <32 x i8> %1)
  ret <4 x i64> %2
}

declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>)

attributes #0 = { "target-cpu"="core2" "target-features"="+ssse3,+cx16,+sse4.2,+sse4.1,+sse,+sse2,+sse3,+avx,+popcnt" }
