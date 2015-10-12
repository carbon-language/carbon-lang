; RUN: llc -mattr=+ssse3 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; The pshufb from function @pr24562 was wrongly folded into its first operand
; as a result of a late target shuffle combine on the legalized selection dag.
; 
; Check that the pshufb is correctly folded to a zero vector.

define <2 x i64> @pr24562() {
; CHECK-LABEL: pr24562:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
entry:
  %0 = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>) #2
  %1 = bitcast <16 x i8> %0 to <2 x i64>
  ret <2 x i64> %1
}

declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>)
