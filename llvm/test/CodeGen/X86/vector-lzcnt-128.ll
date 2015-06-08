; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <2 x i64> @foldv2i64() {
; SSE-LABEL: foldv2i64:
; SSE:       # BB#0:
; SSE-NEXT:    movl $55, %eax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv2i64:
; AVX:       # BB#0:
; AVX-NEXT:    movl $55, %eax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> <i64 256, i64 -1>, i1 0)
  ret <2 x i64> %out
}

define <2 x i64> @foldv2i64u() {
; SSE-LABEL: foldv2i64u:
; SSE:       # BB#0:
; SSE-NEXT:    movl $55, %eax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv2i64u:
; AVX:       # BB#0:
; AVX-NEXT:    movl $55, %eax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> <i64 256, i64 -1>, i1 -1)
  ret <2 x i64> %out
}

define <4 x i32> @foldv4i32() {
; SSE-LABEL: foldv4i32:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [23,0,32,24]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [23,0,32,24]
; AVX-NEXT:    retq
  %out = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> <i32 256, i32 -1, i32 0, i32 255>, i1 0)
  ret <4 x i32> %out
}

define <4 x i32> @foldv4i32u() {
; SSE-LABEL: foldv4i32u:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [23,0,32,24]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv4i32u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [23,0,32,24]
; AVX-NEXT:    retq
  %out = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> <i32 256, i32 -1, i32 0, i32 255>, i1 -1)
  ret <4 x i32> %out
}

define <8 x i16> @foldv8i16() {
; SSE-LABEL: foldv8i16:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [7,0,16,8,16,13,11,9]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [7,0,16,8,16,13,11,9]
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88>, i1 0)
  ret <8 x i16> %out
}

define <8 x i16> @foldv8i16u() {
; SSE-LABEL: foldv8i16u:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [7,0,16,8,16,13,11,9]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv8i16u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [7,0,16,8,16,13,11,9]
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88>, i1 -1)
  ret <8 x i16> %out
}

define <16 x i8> @foldv16i8() {
; SSE-LABEL: foldv16i8:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2]
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32>, i1 0)
  ret <16 x i8> %out
}

define <16 x i8> @foldv16i8u() {
; SSE-LABEL: foldv16i8u:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv16i8u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,8,0,8,5,3,1,0,0,7,6,5,4,3,2]
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32>, i1 -1)
  ret <16 x i8> %out
}

declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1)
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1)
declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16>, i1)
declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8>, i1)
