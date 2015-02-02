; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+mmx | FileCheck %s -check-prefix=X86-32
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | FileCheck %s -check-prefix=X86-64
;
; On Darwin x86-32, v8i8, v4i16, v2i32 values are passed in MM[0-2].
; On Darwin x86-32, v1i64 values are passed in memory.  In this example, they
;                   are never moved into an MM register at all.
; On Darwin x86-64, v8i8, v4i16, v2i32 values are passed in XMM[0-7].
; On Darwin x86-64, v1i64 values are passed in 64-bit GPRs.

@u1 = external global x86_mmx

define void @t1(x86_mmx %v1) nounwind  {
	store x86_mmx %v1, x86_mmx* @u1, align 8
	ret void

; X86-32-LABEL: t1:
; X86-32: movq %mm0

; X86-64-LABEL: t1:
; X86-64: movdq2q %xmm0
; X86-64: movq %mm0
}

@u2 = external global x86_mmx

define void @t2(<1 x i64> %v1) nounwind  {
        %tmp = bitcast <1 x i64> %v1 to x86_mmx
	store x86_mmx %tmp, x86_mmx* @u2, align 8
	ret void

; X86-32-LABEL: t2:
; X86-32: movl 4(%esp)
; X86-32: movl 8(%esp)

; X86-64-LABEL: t2:
; X86-64: movq %rdi
}

@g_v8qi = external global <8 x i8>

define void @t3() nounwind  {
; X86-64-LABEL: t3:
; X86-64-NOT: movdq2q
; X86-64: punpcklbw
  %tmp3 = load <8 x i8>* @g_v8qi, align 8
  %tmp3a = bitcast <8 x i8> %tmp3 to x86_mmx
  %tmp4 = tail call i32 (...)* @pass_v8qi( x86_mmx %tmp3a ) nounwind
  ret void
}

define void @t4(x86_mmx %v1, x86_mmx %v2) nounwind  {
; X86-64-LABEL: t4:
; X86-64: movdq2q
; X86-64: movdq2q
; X86-64-NOT: movdq2q
  %v1a = bitcast x86_mmx %v1 to <8 x i8>
  %v2b = bitcast x86_mmx %v2 to <8 x i8>
  %tmp3 = add <8 x i8> %v1a, %v2b
  %tmp3a = bitcast <8 x i8> %tmp3 to x86_mmx
  %tmp4 = tail call i32 (...)* @pass_v8qi( x86_mmx %tmp3a ) nounwind
  ret void
}

define void @t5() nounwind  {
; X86-64-LABEL: t5:
; X86-64-NOT: movdq2q
; X86-64: xorl  %edi, %edi
  call void @pass_v1di( <1 x i64> zeroinitializer )
  ret void
}

declare i32 @pass_v8qi(...)
declare void @pass_v1di(<1 x i64>)
