; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+sse4.1 | FileCheck %s -check-prefix=SSE41
; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+avx | FileCheck %s -check-prefix=AVX

; CHECK-LABEL: extract_i8
define void @extract_i8(i8* nocapture %dst, <16 x i8> %foo) {
; AVX: vpextrb
; SSE41: pextrb
; AVX-NOT: movb
; SSE41-NOT: movb
  %vecext = extractelement <16 x i8> %foo, i32 15
  store i8 %vecext, i8* %dst, align 1
  ret void
}

; CHECK-LABEL: extract_i16
define void @extract_i16(i16* nocapture %dst, <8 x i16> %foo) {
; AVX: vpextrw
; SSE41: pextrw
; AVX-NOT: movw
; SSE41-NOT: movw
  %vecext = extractelement <8 x i16> %foo, i32 7
  store i16 %vecext, i16* %dst, align 1
  ret void
}

; CHECK-LABEL: extract_i8_undef
define void @extract_i8_undef(i8* nocapture %dst, <16 x i8> %foo) {
; AVX-NOT: vpextrb
; SSE41-NOT: pextrb
; AVX-NOT: movb
; SSE41-NOT: movb
  %vecext = extractelement <16 x i8> %foo, i32 16 ; undef
  store i8 %vecext, i8* %dst, align 1
  ret void
}

; CHECK-LABEL: extract_i16_undef
define void @extract_i16_undef(i16* nocapture %dst, <8 x i16> %foo) {
; AVX-NOT: vpextrw
; SSE41-NOT: pextrw
; AVX-NOT: movw
; SSE41-NOT: movw
  %vecext = extractelement <8 x i16> %foo, i32 9 ; undef
  store i16 %vecext, i16* %dst, align 1
  ret void
}
