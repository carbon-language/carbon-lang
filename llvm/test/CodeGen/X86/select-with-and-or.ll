; RUN: opt < %s -O3 | \
; RUN:	llc -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

define <4 x i32> @test1(<4 x float> %a, <4 x float> %b, <4 x i32> %c) {
  %f = fcmp ult <4 x float> %a, %b
  %r = select <4 x i1> %f, <4 x i32> %c, <4 x i32> zeroinitializer
  ret <4 x i32> %r
; CHECK: test1
; CHECK: cmpnle
; CHECK-NEXT: andps
; CHECK: ret
}

define <4 x i32> @test2(<4 x float> %a, <4 x float> %b, <4 x i32> %c) {
  %f = fcmp ult <4 x float> %a, %b
  %r = select <4 x i1> %f, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c
  ret <4 x i32> %r
; CHECK: test2
; CHECK: cmpnle
; CHECK-NEXT: orps
; CHECK: ret
}

define <4 x i32> @test3(<4 x float> %a, <4 x float> %b, <4 x i32> %c) {
  %f = fcmp ult <4 x float> %a, %b
  %r = select <4 x i1> %f, <4 x i32> zeroinitializer, <4 x i32> %c
  ret <4 x i32> %r
; CHECK: test3
; CHECK: cmple
; CHECK-NEXT: andps
; CHECK: ret
}

define <4 x i32> @test4(<4 x float> %a, <4 x float> %b, <4 x i32> %c) {
  %f = fcmp ult <4 x float> %a, %b
  %r = select <4 x i1> %f, <4 x i32> %c, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %r
; CHECK: test4
; CHECK: cmple
; CHECK-NEXT: orps
; CHECK: ret
}

define <4 x i32> @test5(<4 x float> %a, <4 x float> %b, <4 x i32> %c) {
  %f = fcmp ult <4 x float> %a, %b
  %r = select <4 x i1> %f, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer
  ret <4 x i32> %r
; CHECK: test5
; CHECK: cmpnle
; CHECK-NEXT: ret
}

define <4 x i32> @test6(<4 x float> %a, <4 x float> %b, <4 x i32> %c) {
  %f = fcmp ult <4 x float> %a, %b
  %r = select <4 x i1> %f, <4 x i32> zeroinitializer, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %r
; CHECK: test6
; CHECK: cmple
; CHECK-NEXT: ret
}

define <4 x i32> @test7(<4 x float> %a, <4 x float> %b, <4 x i32>* %p) {
  %f = fcmp ult <4 x float> %a, %b
  %s = sext <4 x i1> %f to <4 x i32>
  %l = load <4 x i32>* %p
  %r = and <4 x i32> %l, %s
  ret <4 x i32> %r
; CHECK: test7
; CHECK: cmpnle
; CHECK-NEXT: andps
; CHECK: ret
}
