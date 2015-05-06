; RUN: llc < %s -mtriple=x86_64-pc-linux             | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-pc-linux -mattr=+avx | FileCheck %s -check-prefix=AVX


; CHECK-LABEL: extractelement_index_1:
define i8 @extractelement_index_1(<32 x i8> %a) nounwind {
  ; X64:       movaps
  ; AVX:       vpextrb $1
  %b = extractelement <32 x i8> %a, i256 1
  ret i8 %b
}

; CHECK-LABEL: extractelement_index_2:
define i32 @extractelement_index_2(<8 x i32> %a) nounwind {
  ; X64:       pshufd
  ; AVX:       vextractf128 $1
  ; AVX-NEXT:  vpextrd      $3
  %b = extractelement <8 x i32> %a, i64 7
  ret i32 %b
}

; CHECK-LABEL: extractelement_index_3:
define i32 @extractelement_index_3(<8 x i32> %a) nounwind {
  ; CHECK-NOT: pextr
  %b = extractelement <8 x i32> %a, i64 15
  ret i32 %b
}

; CHECK-LABEL: extractelement_index_4:
define i32 @extractelement_index_4(<8 x i32> %a) nounwind {
  ; X64:       movd
  ; AVX:       vextractf128 $1
  ; AVX-NEXT:  vmovd
  %b = extractelement <8 x i32> %a, i256 4
  ret i32 %b
}

; CHECK-LABEL: extractelement_index_5:
define i8 @extractelement_index_5(<32 x i8> %a, i256 %i) nounwind {
  ; X64:       movaps
  ; AVX:       vmovaps
  %b = extractelement <32 x i8> %a, i256 %i
  ret i8 %b
}

; CHECK-LABEL: extractelement_index_6:
define i8 @extractelement_index_6(<32 x i8> %a) nounwind {
  ; CHECK-NOT: pextr
  %b = extractelement <32 x i8> %a, i256 -1
  ret i8 %b
}