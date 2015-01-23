; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mtriple=x86_64-pc-win32 | FileCheck %s

; CHECK-LABEL: vcast:
define <2 x i32> @vcast(<2 x float> %a, <2 x float> %b) {
; CHECK-NOT: pmovzxdq
; CHECK-NOT: pmovzxdq
; CHECK: movdqa (%{{.*}}),  %[[R0:xmm[0-9]+]]
  %af = bitcast <2 x float> %a to <2 x i32>
  %bf = bitcast <2 x float> %b to <2 x i32>
; CHECK-NEXT: psubq (%{{.*}}), %[[R0]]
  %x = sub <2 x i32> %af, %bf
; CHECK: ret
  ret <2 x i32> %x
}

