; Three variants of "MSVC" environments.
; RUN: llc < %s -mattr=sse2 -mattr=avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=sse2 -mattr=avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-windows-msvc -mattr=sse2 -mattr=avx | FileCheck %s
; GNU environment.
; RUN: llc < %s -mtriple=x86_64-win32-gnu -mattr=sse2 -mattr=avx | FileCheck -check-prefix=MINGW %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-win32"

define double @double() {
  ret double 0x0000000000800000
}
; CHECK:              .globl  __real@0000000000800000
; CHECK-NEXT:         .section        .rdata,"dr",discard,__real@0000000000800000
; CHECK-NEXT:         .p2align  3
; CHECK-NEXT: __real@0000000000800000:
; CHECK-NEXT:         .quad   0x0000000000800000
; CHECK:      double:
; CHECK:               movsd   __real@0000000000800000(%rip), %xmm0
; CHECK-NEXT:          ret

; MINGW:              .section        .rdata,"dr"
; MINGW-NEXT:         .p2align  3
; MINGW-NEXT: [[LABEL:\.LC.*]]:
; MINGW-NEXT:         .quad   0x0000000000800000
; MINGW:      double:
; MINGW:               movsd   [[LABEL]](%rip), %xmm0
; MINGW-NEXT:          ret

define <4 x i32> @vec1() {
  ret <4 x i32> <i32 3, i32 2, i32 1, i32 0>
}
; CHECK:              .globl  __xmm@00000000000000010000000200000003
; CHECK-NEXT:         .section        .rdata,"dr",discard,__xmm@00000000000000010000000200000003
; CHECK-NEXT:         .p2align  4
; CHECK-NEXT: __xmm@00000000000000010000000200000003:
; CHECK-NEXT:         .long   3
; CHECK-NEXT:         .long   2
; CHECK-NEXT:         .long   1
; CHECK-NEXT:         .long   0
; CHECK:      vec1:
; CHECK:               movaps  __xmm@00000000000000010000000200000003(%rip), %xmm0
; CHECK-NEXT:          ret

define <8 x i16> @vec2() {
  ret <8 x i16> <i16 7, i16 6, i16 5, i16 4, i16 3, i16 2, i16 1, i16 0>
}
; CHECK:             .globl  __xmm@00000001000200030004000500060007
; CHECK-NEXT:        .section        .rdata,"dr",discard,__xmm@00000001000200030004000500060007
; CHECK-NEXT:        .p2align  4
; CHECK-NEXT: __xmm@00000001000200030004000500060007:
; CHECK-NEXT:        .short  7
; CHECK-NEXT:        .short  6
; CHECK-NEXT:        .short  5
; CHECK-NEXT:        .short  4
; CHECK-NEXT:        .short  3
; CHECK-NEXT:        .short  2
; CHECK-NEXT:        .short  1
; CHECK-NEXT:        .short  0
; CHECK:      vec2:
; CHECK:               movaps  __xmm@00000001000200030004000500060007(%rip), %xmm0
; CHECK-NEXT:          ret


define <4 x float> @undef1() {
  ret <4 x float> <float 1.0, float 1.0, float undef, float undef>

; CHECK:             .globl  __xmm@00000000000000003f8000003f800000
; CHECK-NEXT:        .section        .rdata,"dr",discard,__xmm@00000000000000003f8000003f800000
; CHECK-NEXT:        .p2align  4
; CHECK-NEXT: __xmm@00000000000000003f8000003f800000:
; CHECK-NEXT:        .long   0x3f800000              # float 1
; CHECK-NEXT:        .long   0x3f800000              # float 1
; CHECK-NEXT:        .zero   4
; CHECK-NEXT:        .zero   4
; CHECK:      undef1:
; CHECK:               movaps  __xmm@00000000000000003f8000003f800000(%rip), %xmm0
; CHECK-NEXT:          ret
}

define float @pr23966(i32 %a) {
  %tobool = icmp ne i32 %a, 0
  %sel = select i1 %tobool, float -1.000000e+00, float 1.000000e+00
  ret float %sel
}

; CHECK:              .globl  __real@bf8000003f800000
; CHECK-NEXT:         .section        .rdata,"dr",discard,__real@bf8000003f800000
; CHECK-NEXT:         .p2align  3
; CHECK-NEXT: __real@bf8000003f800000:
; CHECK-NEXT:         .long   0x3f800000
; CHECK-NEXT:         .long   0xbf800000

define <4 x i64> @ymm() {
entry:
  ret <4 x i64> <i64 8589934593, i64 17179869187, i64 8589934593, i64 17179869187>
}

; CHECK:	.globl	__ymm@0000000400000003000000020000000100000004000000030000000200000001
; CHECK:	.section	.rdata,"dr",discard,__ymm@0000000400000003000000020000000100000004000000030000000200000001
; CHECK:	.p2align	5
; CHECK: __ymm@0000000400000003000000020000000100000004000000030000000200000001:
; CHECK: 	.quad	8589934593              # 0x200000001
; CHECK: 	.quad	17179869187             # 0x400000003
; CHECK: 	.quad	8589934593              # 0x200000001
; CHECK: 	.quad	17179869187
