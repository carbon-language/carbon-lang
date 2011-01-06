; RUN: llc < %s -mattr=+sse2      -mtriple=i686-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -mattr=-sse       -mtriple=i686-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=NOSSE
; RUN: llc < %s                 -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s -check-prefix=X86-64

@.str = internal constant [25 x i8] c"image\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
@.str2 = internal constant [30 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 4

define void @t1(i32 %argc, i8** %argv) nounwind  {
entry:
; SSE2: t1:
; SSE2: movaps _.str, %xmm0
; SSE2: movaps %xmm0
; SSE2: movb $0
; SSE2: movl $0
; SSE2: movl $0

; NOSSE: t1:
; NOSSE: movb $0
; NOSSE: movl $0
; NOSSE: movl $0
; NOSSE: movl $0
; NOSSE: movl $0
; NOSSE: movl $101
; NOSSE: movl $1734438249

; X86-64: t1:
; X86-64: movaps _.str(%rip), %xmm0
; X86-64: movaps %xmm0
; X86-64: movb $0
; X86-64: movq $0
  %tmp1 = alloca [25 x i8]
  %tmp2 = bitcast [25 x i8]* %tmp1 to i8*
  call void @llvm.memcpy.i32( i8* %tmp2, i8* getelementptr ([25 x i8]* @.str, i32 0, i32 0), i32 25, i32 1 ) nounwind 
  unreachable
}

;rdar://7774704
%struct.s0 = type { [2 x double] }

define void @t2(%struct.s0* nocapture %a, %struct.s0* nocapture %b) nounwind ssp {
entry:
; SSE2: t2:
; SSE2: movaps (%eax), %xmm0
; SSE2: movaps %xmm0, (%eax)

; NOSSE: t2:
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl

; X86-64: t2:
; X86-64: movaps (%rsi), %xmm0
; X86-64: movaps %xmm0, (%rdi)
  %tmp2 = bitcast %struct.s0* %a to i8*           ; <i8*> [#uses=1]
  %tmp3 = bitcast %struct.s0* %b to i8*           ; <i8*> [#uses=1]
  tail call void @llvm.memcpy.i32(i8* %tmp2, i8* %tmp3, i32 16, i32 16)
  ret void
}

define void @t3(%struct.s0* nocapture %a, %struct.s0* nocapture %b) nounwind ssp {
entry:
; SSE2: t3:
; SSE2: movups (%eax), %xmm0
; SSE2: movups %xmm0, (%eax)

; NOSSE: t3:
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl
; NOSSE: movl

; X86-64: t3:
; X86-64: movups (%rsi), %xmm0
; X86-64: movups %xmm0, (%rdi)
  %tmp2 = bitcast %struct.s0* %a to i8*           ; <i8*> [#uses=1]
  %tmp3 = bitcast %struct.s0* %b to i8*           ; <i8*> [#uses=1]
  tail call void @llvm.memcpy.i32(i8* %tmp2, i8* %tmp3, i32 16, i32 8)
  ret void
}

define void @t4() nounwind {
entry:
; SSE2: t4:
; SSE2: movups _.str2, %xmm0
; SSE2: movaps %xmm0, (%esp)
; SSE2: movw $120, 28(%esp)
; SSE2: movl $2021161080
; SSE2: movl $2021161080
; SSE2: movl $2021161080

; NOSSE: t4:
; NOSSE: movw $120
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080
; NOSSE: movl $2021161080

; X86-64: t4:
; X86-64: movabsq $8680820740569200760, %rax
; X86-64: movq %rax
; X86-64: movups _.str2(%rip), %xmm0
; X86-64: movaps %xmm0, -40(%rsp)
; X86-64: movw $120
; X86-64: movl $2021161080
  %tmp1 = alloca [30 x i8]
  %tmp2 = bitcast [30 x i8]* %tmp1 to i8*
  call void @llvm.memcpy.i32(i8* %tmp2, i8* getelementptr inbounds ([30 x i8]* @.str2, i32 0, i32 0), i32 30, i32 1)
  unreachable
}

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind
