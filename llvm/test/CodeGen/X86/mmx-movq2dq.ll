; RUN: llc < %s -march=x86 -mattr=+mmx,+sse2 | FileCheck %s -check-prefix=X86-32
; RUN: llc < %s -march=x86-64 -mattr=+mmx,+sse2 | FileCheck %s -check-prefix=X86-64

; X86-32-LABEL: test0
; X86-64-LABEL: test0
define i32 @test0(<1 x i64>* %v4) {
  %v5 = load <1 x i64>* %v4, align 8
  %v12 = bitcast <1 x i64> %v5 to <4 x i16>
  %v13 = bitcast <4 x i16> %v12 to x86_mmx
  ; X86-32: pshufw  $238
  ; X86-32-NOT: movq
  ; X86-32-NOT: movsd
  ; X86-32: movq2dq
  ; X86-64: pshufw  $238
  ; X86-64-NOT: movq
  ; X86-64-NOT: pshufd
  ; X86-64: movq2dq
  ; X86-64-NEXT: movd
  %v14 = tail call x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx %v13, i8 -18)
  %v15 = bitcast x86_mmx %v14 to <4 x i16>
  %v16 = bitcast <4 x i16> %v15 to <1 x i64>
  %v17 = extractelement <1 x i64> %v16, i32 0
  %v18 = bitcast i64 %v17 to <2 x i32>
  %v19 = extractelement <2 x i32> %v18, i32 0
  %v20 = add i32 %v19, 32
  ret i32 %v20
}

declare x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx, i8)
