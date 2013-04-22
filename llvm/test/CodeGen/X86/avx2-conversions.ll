; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: trunc4
; CHECK: vpermd
; CHECK-NOT: vinsert
; CHECK: ret
define <4 x i32> @trunc4(<4 x i64> %A) nounwind {
  %B = trunc <4 x i64> %A to <4 x i32>
  ret <4 x i32>%B
}

; CHECK: trunc8
; CHECK: vpshufb
; CHECK-NOT: vinsert
; CHECK: ret

define <8 x i16> @trunc8(<8 x i32> %A) nounwind {
  %B = trunc <8 x i32> %A to <8 x i16>
  ret <8 x i16>%B
}

; CHECK: sext4
; CHECK: vpmovsxdq
; CHECK-NOT: vinsert
; CHECK: ret
define <4 x i64> @sext4(<4 x i32> %A) nounwind {
  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

; CHECK: sext8
; CHECK: vpmovsxwd
; CHECK-NOT: vinsert
; CHECK: ret
define <8 x i32> @sext8(<8 x i16> %A) nounwind {
  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

; CHECK: zext4
; CHECK: vpmovzxdq
; CHECK-NOT: vinsert
; CHECK: ret
define <4 x i64> @zext4(<4 x i32> %A) nounwind {
  %B = zext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

; CHECK: zext8
; CHECK: vpmovzxwd
; CHECK-NOT: vinsert
; CHECK: ret
define <8 x i32> @zext8(<8 x i16> %A) nounwind {
  %B = zext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}
; CHECK: zext_8i8_8i32
; CHECK: vpmovzxwd
; CHECK: vpand
; CHECK: ret
define <8 x i32> @zext_8i8_8i32(<8 x i8> %A) nounwind {
  %B = zext <8 x i8> %A to <8 x i32>  
  ret <8 x i32>%B
}

; CHECK: load_sext_test1
; CHECK: vpmovsxdq (%r{{[^,]*}}), %ymm{{.*}}
; CHECK: ret 
define <4 x i64> @load_sext_test1(<4 x i32> *%ptr) {
 %X = load <4 x i32>* %ptr
 %Y = sext <4 x i32> %X to <4 x i64>
 ret <4 x i64>%Y
}

; CHECK: load_sext_test2
; CHECK: vpmovsxbq (%r{{[^,]*}}), %ymm{{.*}}
; CHECK: ret 
define <4 x i64> @load_sext_test2(<4 x i8> *%ptr) {
 %X = load <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i64>
 ret <4 x i64>%Y
}

; CHECK: load_sext_test3
; CHECK: vpmovsxwq (%r{{[^,]*}}), %ymm{{.*}}
; CHECK: ret 
define <4 x i64> @load_sext_test3(<4 x i16> *%ptr) {
 %X = load <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i64>
 ret <4 x i64>%Y
}

; CHECK: load_sext_test4
; CHECK: vpmovsxwd (%r{{[^,]*}}), %ymm{{.*}}
; CHECK: ret 
define <8 x i32> @load_sext_test4(<8 x i16> *%ptr) {
 %X = load <8 x i16>* %ptr
 %Y = sext <8 x i16> %X to <8 x i32>
 ret <8 x i32>%Y
}

; CHECK: load_sext_test5
; CHECK: vpmovsxbd (%r{{[^,]*}}), %ymm{{.*}}
; CHECK: ret 
define <8 x i32> @load_sext_test5(<8 x i8> *%ptr) {
 %X = load <8 x i8>* %ptr
 %Y = sext <8 x i8> %X to <8 x i32>
 ret <8 x i32>%Y
}
