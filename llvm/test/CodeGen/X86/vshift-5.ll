; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -disable-mmx | FileCheck %s

; When loading the shift amount from memory, avoid generating the splat.

define void @shift5a(<4 x i32> %val, <4 x i32>* %dst, i32* %pamt) nounwind {
entry:
; CHECK: shift5a:
; CHECK: movd
; CHECK-NEXT: pslld
  %amt = load i32* %pamt 
  %tmp0 = insertelement <4 x i32> undef, i32 %amt, i32 0
  %shamt = shufflevector <4 x i32> %tmp0, <4 x i32> undef, <4 x i32> zeroinitializer 
  %shl = shl <4 x i32> %val, %shamt
  store <4 x i32> %shl, <4 x i32>* %dst
  ret void
}


define void @shift5b(<4 x i32> %val, <4 x i32>* %dst, i32* %pamt) nounwind {
entry:
; CHECK: shift5b:
; CHECK: movd
; CHECK-NEXT: psrad
  %amt = load i32* %pamt 
  %tmp0 = insertelement <4 x i32> undef, i32 %amt, i32 0
  %shamt = shufflevector <4 x i32> %tmp0, <4 x i32> undef, <4 x i32> zeroinitializer 
  %shr = ashr <4 x i32> %val, %shamt
  store <4 x i32> %shr, <4 x i32>* %dst
  ret void
}


define void @shift5c(<4 x i32> %val, <4 x i32>* %dst, i32 %amt) nounwind {
entry:
; CHECK: shift5c:
; CHECK: movd
; CHECK-NEXT: pslld
  %tmp0 = insertelement <4 x i32> undef, i32 %amt, i32 0
  %shamt = shufflevector <4 x i32> %tmp0, <4 x i32> undef, <4 x i32> zeroinitializer
  %shl = shl <4 x i32> %val, %shamt
  store <4 x i32> %shl, <4 x i32>* %dst
  ret void
}


define void @shift5d(<4 x i32> %val, <4 x i32>* %dst, i32 %amt) nounwind {
entry:
; CHECK: shift5d:
; CHECK: movd
; CHECK-NEXT: psrad
  %tmp0 = insertelement <4 x i32> undef, i32 %amt, i32 0
  %shamt = shufflevector <4 x i32> %tmp0, <4 x i32> undef, <4 x i32> zeroinitializer
  %shr = ashr <4 x i32> %val, %shamt
  store <4 x i32> %shr, <4 x i32>* %dst
  ret void
}
