; RUN: llc < %s -march=x86-64 -mattr=+sse2,-sse4.1 | FileCheck %s

define <2 x i64> @t1(i64 %s, <2 x i64> %tmp) nounwind {
; CHECK-LABEL: t1:
; CHECK:  punpcklqdq 
; CHECK-NEXT:  retq 

  %tmp1 = insertelement <2 x i64> %tmp, i64 %s, i32 1
  ret <2 x i64> %tmp1
}
