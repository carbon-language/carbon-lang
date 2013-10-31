; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl  | FileCheck %s

; CHECK-LABEL: select00
; CHECK: vmovaps
; CHECK-NEXT: LBB
define <16 x i32> @select00(i32 %a, <16 x i32> %b) nounwind {
  %cmpres = icmp eq i32 %a, 255
  %selres = select i1 %cmpres, <16 x i32> zeroinitializer, <16 x i32> %b
  %res = xor <16 x i32> %b, %selres
  ret <16 x i32> %res
}

; CHECK-LABEL: select01
; CHECK: vmovaps
; CHECK-NEXT: LBB
define <8 x i64> @select01(i32 %a, <8 x i64> %b) nounwind {
  %cmpres = icmp eq i32 %a, 255
  %selres = select i1 %cmpres, <8 x i64> zeroinitializer, <8 x i64> %b
  %res = xor <8 x i64> %b, %selres
  ret <8 x i64> %res
}

