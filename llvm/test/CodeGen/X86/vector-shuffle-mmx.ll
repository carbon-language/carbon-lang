; RUN: llc < %s -march=x86 -mattr=+mmx,+sse2 | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -march=x86-64 -mattr=+mmx,+sse2 | FileCheck -check-prefix=X64 %s

; If there is no explicit MMX type usage, always promote to XMM.

define void @test0(<1 x i64>* %x) {
; X32-LABEL: test0
; X64-LABEL: test0
; X32: pshufd $213
; X64: pshufd $213
; X32-NEXT: movlpd %xmm
; X64-NEXT: movq %xmm
entry:
  %tmp2 = load <1 x i64>* %x
  %tmp6 = bitcast <1 x i64> %tmp2 to <2 x i32>
  %tmp9 = shufflevector <2 x i32> %tmp6, <2 x i32> undef, <2 x i32> < i32 1, i32 1 >
  %tmp10 = bitcast <2 x i32> %tmp9 to <1 x i64>
  store <1 x i64> %tmp10, <1 x i64>* %x
  ret void
}

define void @test1() {
; X32-LABEL: test1:
; X32:    pshuflw
; X32-NEXT:    pshufhw
; X32-NEXT:    pshufd
; X32:    maskmovq
entry:
  %tmp528 = bitcast <8 x i8> zeroinitializer to <2 x i32>
  %tmp529 = and <2 x i32> %tmp528, bitcast (<4 x i16> < i16 -32640, i16 16448, i16 8224, i16 4112 > to <2 x i32>)
  %tmp542 = bitcast <2 x i32> %tmp529 to <4 x i16>
  %tmp543 = add <4 x i16> %tmp542, < i16 0, i16 16448, i16 24672, i16 28784 >
  %tmp555 = bitcast <4 x i16> %tmp543 to <8 x i8>
  %tmp556 = bitcast <8 x i8> %tmp555 to x86_mmx
  %tmp557 = bitcast <8 x i8> zeroinitializer to x86_mmx
  tail call void @llvm.x86.mmx.maskmovq( x86_mmx %tmp557, x86_mmx %tmp556, i8* null)
  ret void
}

declare void @llvm.x86.mmx.maskmovq(x86_mmx, x86_mmx, i8*)
