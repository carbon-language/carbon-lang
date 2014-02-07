; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+sse4.1 | FileCheck %s -check-prefix=SSE41
; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+avx | FileCheck %s -check-prefix=AVX

define void @pextrb(i8* nocapture %dst, <16 x i8> %foo) {
; AVX: vpextrb
; SSE41: pextrb
; AVX-NOT: movb
; SSE41-NOT: movb
  %vecext = extractelement <16 x i8> %foo, i32 15
  store i8 %vecext, i8* %dst, align 1
  ret void
}

define void @pextrw(i16* nocapture %dst, <8 x i16> %foo) {
; AVX: vpextrw
; SSE41: pextrw
; AVX-NOT: movw
; SSE41-NOT: movw
  %vecext = extractelement <8 x i16> %foo, i32 15
  store i16 %vecext, i16* %dst, align 1
  ret void
}
