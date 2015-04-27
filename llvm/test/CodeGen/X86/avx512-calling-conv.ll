; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=SKX

; KNL-LABEL: test1
; KNL: vxorps
define <16 x i1> @test1() {
  ret <16 x i1> zeroinitializer
}

; SKX-LABEL: test2
; SKX: vpmovb2m
; SKX: vpmovb2m
; SKX: kandw
; SKX: vpmovm2b
; KNL-LABEL: test2
; KNL: vpmovsxbd
; KNL: vpmovsxbd
; KNL: vpandd
; KNL: vpmovdb
define <16 x i1> @test2(<16 x i1>%a, <16 x i1>%b) {
  %c = and <16 x i1>%a, %b
  ret <16 x i1> %c
}

; SKX-LABEL: test3
; SKX: vpmovw2m
; SKX: vpmovw2m
; SKX: kandb
; SKX: vpmovm2w
define <8 x i1> @test3(<8 x i1>%a, <8 x i1>%b) {
  %c = and <8 x i1>%a, %b
  ret <8 x i1> %c
}

; SKX-LABEL: test4
; SKX: vpmovd2m
; SKX: vpmovd2m
; SKX: kandw
; SKX: vpmovm2d
define <4 x i1> @test4(<4 x i1>%a, <4 x i1>%b) {
  %c = and <4 x i1>%a, %b
  ret <4 x i1> %c
}

