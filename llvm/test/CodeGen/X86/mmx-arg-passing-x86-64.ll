; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+mmx,+sse2 | FileCheck %s --check-prefix=X86-64
;
; On Darwin x86-64, v8i8, v4i16, v2i32 values are passed in XMM[0-7].
; On Darwin x86-64, v1i64 values are passed in 64-bit GPRs.

@g_v8qi = external global <8 x i8>

define void @t3() nounwind  {
; X86-64-LABEL: t3:
; X86-64:       ## BB#0:
; X86-64-NEXT:    movq _g_v8qi@{{.*}}(%rip), %rax
; X86-64-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; X86-64-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; X86-64-NEXT:    pand {{.*}}(%rip), %xmm0
; X86-64-NEXT:    packuswb %xmm0, %xmm0
; X86-64-NEXT:    movd %xmm0, %rax
; X86-64-NEXT:    movd %rax, %xmm0
; X86-64-NEXT:    movb $1, %al
; X86-64-NEXT:    jmp _pass_v8qi ## TAILCALL
  %tmp3 = load <8 x i8>* @g_v8qi, align 8
  %tmp3a = bitcast <8 x i8> %tmp3 to x86_mmx
  %tmp4 = tail call i32 (...)* @pass_v8qi( x86_mmx %tmp3a ) nounwind
  ret void
}

define void @t4(x86_mmx %v1, x86_mmx %v2) nounwind  {
; X86-64-LABEL: t4:
; X86-64:       ## BB#0:
; X86-64-NEXT:    movdq2q %xmm1, %mm0
; X86-64-NEXT:    movq %mm0, -{{[0-9]+}}(%rsp)
; X86-64-NEXT:    movdq2q %xmm0, %mm0
; X86-64-NEXT:    movq %mm0, -{{[0-9]+}}(%rsp)
; X86-64-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; X86-64-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; X86-64-NEXT:    paddb %xmm0, %xmm1
; X86-64-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; X86-64-NEXT:    pand {{.*}}(%rip), %xmm1
; X86-64-NEXT:    packuswb %xmm1, %xmm1
; X86-64-NEXT:    movd %xmm1, %rax
; X86-64-NEXT:    movd %rax, %xmm0
; X86-64-NEXT:    movb $1, %al
; X86-64-NEXT:    jmp _pass_v8qi ## TAILCALL
  %v1a = bitcast x86_mmx %v1 to <8 x i8>
  %v2b = bitcast x86_mmx %v2 to <8 x i8>
  %tmp3 = add <8 x i8> %v1a, %v2b
  %tmp3a = bitcast <8 x i8> %tmp3 to x86_mmx
  %tmp4 = tail call i32 (...)* @pass_v8qi( x86_mmx %tmp3a ) nounwind
  ret void
}

define void @t5() nounwind  {
; X86-64-LABEL: t5:
; X86-64:       ## BB#0:
; X86-64-NEXT:    pushq %rax
; X86-64-NEXT:    xorl %edi, %edi
; X86-64-NEXT:    callq _pass_v1di
; X86-64-NEXT:    popq %rax
; X86-64-NEXT:    retq
  call void @pass_v1di( <1 x i64> zeroinitializer )
  ret void
}

declare i32 @pass_v8qi(...)
declare void @pass_v1di(<1 x i64>)
