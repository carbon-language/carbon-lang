; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1

define i64 @trunc2i64(<2 x i64> %inval) {
; SSE-LABEL:  trunc2i64:
; SSE:        # BB#0: # %entry
; SSE-NEXT:   pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSE-NEXT:   movd %xmm0, %rax
; SSE-NEXT:   retq

; AVX-LABEL:  trunc2i64:
; AVX:        # BB#0: # %entry
; AVX-NEXT:   vpshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; AVX-NEXT:   vmovq %xmm0, %rax
; AVX-NEXT:   retq

entry:
  %0 = trunc <2 x i64> %inval to <2 x i32>
  %1 = bitcast <2 x i32> %0 to i64
  ret i64 %1
}

; PR15524 http://llvm.org/bugs/show_bug.cgi?id=15524
define i64 @trunc4i32(<4 x i32> %inval) {
; SSE2-LABEL:  trunc4i32:
; SSE2:        # BB#0: # %entry
; SSE2-NEXT:   pshuflw {{.*#+}} xmm0 = xmm0[0,2,2,3,4,5,6,7]
; SSE2-NEXT:   pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,6,6,7]
; SSE2-NEXT:   pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:   movd %xmm0, %rax
; SSE2-NEXT:   retq

; SSSE3-LABEL: trunc4i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:  pshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; SSSE3-NEXT:  movd %xmm0, %rax
; SSSE3-NEXT:  retq

; SSE41-LABEL: trunc4i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:  pshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; SSE41-NEXT:  movd %xmm0, %rax
; SSE41-NEXT:  retq

; AVX-LABEL:  trunc4i32:
; AVX:        # BB#0: # %entry
; AVX-NEXT:   vpshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; AVX-NEXT:   vmovq %xmm0, %rax
; AVX-NEXT:   retq

entry:
  %0 = trunc <4 x i32> %inval to <4 x i16>
  %1 = bitcast <4 x i16> %0 to i64
  ret i64 %1
}

; PR15524 http://llvm.org/bugs/show_bug.cgi?id=15524
define i64 @trunc8i16(<8 x i16> %inval) {
; SSE2-LABEL:  trunc8i16:
; SSE2:        # BB#0: # %entry
; SSE2-NEXT:   pand .LCP{{.*}}(%rip), %xmm0
; SSE2-NEXT:   packuswb %xmm0, %xmm0
; SSE2-NEXT:   movd %xmm0, %rax
; SSE2-NEXT:   retq

; SSSE3-LABEL: trunc8i16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:  pshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; SSSE3-NEXT:  movd %xmm0, %rax
; SSSE3-NEXT:  retq

; SSE41-LABEL: trunc8i16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:  pshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; SSE41-NEXT:  movd %xmm0, %rax
; SSE41-NEXT:  retq

; AVX-LABEL:  trunc8i16:
; AVX:        # BB#0: # %entry
; AVX-NEXT:   vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; AVX-NEXT:   vmovq %xmm0, %rax
; AVX-NEXT:   retq

entry:
  %0 = trunc <8 x i16> %inval to <8 x i8>
  %1 = bitcast <8 x i8> %0 to i64
  ret i64 %1
}
