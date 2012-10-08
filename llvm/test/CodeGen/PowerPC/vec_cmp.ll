; RUN: llc -mattr=+altivec < %s | FileCheck %s

; Check vector comparisons using altivec.


target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <2 x i8> @v2si8_cmp(<2 x i8> %x, <2 x i8> %y) nounwind readnone {
  %cmp = icmp eq <2 x i8> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %sext
}
; CHECK: v2si8_cmp:
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <4 x i8> @v4si8_cmp(<4 x i8> %x, <4 x i8> %y) nounwind readnone {
  %cmp = icmp eq <4 x i8> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %sext
}
; CHECK: v4si8_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <8 x i8> @v8si8_cmp(<8 x i8> %x, <8 x i8> %y) nounwind readnone {
  %cmp = icmp eq <8 x i8> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %sext
}
; CHECK: v8si8_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <16 x i8> @v16si8_cmp(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
  %cmp = icmp eq <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK: v16si8_cmp:
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <32 x i8> @v32si8_cmp(<32 x i8> %x, <32 x i8> %y) nounwind readnone {
  %cmp = icmp eq <32 x i8> %x, %y
  %sext = sext <32 x i1> %cmp to <32 x i8>
  ret <32 x i8> %sext
}
; CHECK: v32si8_cmp:
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <2 x i16> @v2si16_cmp(<2 x i16> %x, <2 x i16> %y) nounwind readnone {
  %cmp = icmp eq <2 x i16> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %sext
}
; CHECK: v2si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <4 x i16> @v4si16_cmp(<4 x i16> %x, <4 x i16> %y) nounwind readnone {
  %cmp = icmp eq <4 x i16> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %sext
}
; CHECK: v4si16_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <8 x i16> @v8si16_cmp(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
  %cmp = icmp eq <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK: v8si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <16 x i16> @v16si16_cmp(<16 x i16> %x, <16 x i16> %y) nounwind readnone {
  %cmp = icmp eq <16 x i16> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %sext
}
; CHECK: v16si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <32 x i16> @v32si16_cmp(<32 x i16> %x, <32 x i16> %y) nounwind readnone {
  %cmp = icmp eq <32 x i16> %x, %y
  %sext = sext <32 x i1> %cmp to <32 x i16>
  ret <32 x i16> %sext
}
; CHECK: v32si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <2 x i32> @v2si32_cmp(<2 x i32> %x, <2 x i32> %y) nounwind readnone {
  %cmp = icmp eq <2 x i32> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %sext
}
; CHECK: v2si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <4 x i32> @v4si32_cmp(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
  %cmp = icmp eq <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK: v4si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <8 x i32> @v8si32_cmp(<8 x i32> %x, <8 x i32> %y) nounwind readnone {
  %cmp = icmp eq <8 x i32> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %sext
}
; CHECK: v8si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <16 x i32> @v16si32_cmp(<16 x i32> %x, <16 x i32> %y) nounwind readnone {
  %cmp = icmp eq <16 x i32> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %sext
}
; CHECK: v16si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <32 x i32> @v32si32_cmp(<32 x i32> %x, <32 x i32> %y) nounwind readnone {
  %cmp = icmp eq <32 x i32> %x, %y
  %sext = sext <32 x i1> %cmp to <32 x i32>
  ret <32 x i32> %sext
}
; CHECK: v32si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <2 x float> @v2f32_cmp(<2 x float> %x, <2 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oeq <2 x float> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i32>
  %0 = bitcast <2 x i32> %sext to <2 x float>
  ret <2 x float> %0
}
; CHECK: v2f32_cmp:
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <4 x float> @v4f32_cmp(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oeq <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK: v4f32_cmp:
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <8 x float> @v8f32_cmp(<8 x float> %x, <8 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oeq <8 x float> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i32>
  %0 = bitcast <8 x i32> %sext to <8 x float>
  ret <8 x float> %0
}
; CHECK: v8f32_cmp:
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
