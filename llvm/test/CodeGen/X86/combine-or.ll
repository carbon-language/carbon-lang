; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s


; Verify that each of the following test cases is folded into a single
; instruction which performs a blend operation.

define <2 x i64> @test1(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 1>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}


define <4 x i32> @test2(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test2:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <2 x i64> @test3(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test3:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 1>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}


define <4 x i32> @test4(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test4:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm0[0,1],xmm1[2,3,4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 1, i32 2, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test5:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5,6,7]
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 1, i32 2, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <4 x i32> @test6(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test6:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <4 x i32> @test7(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test7:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; CHECK-NEXT:    retq
  %and1 = and <4 x i32> %a, <i32 -1, i32 -1, i32 0, i32 0>
  %and2 = and <4 x i32> %b, <i32 0, i32 0, i32 -1, i32 -1>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}


define <2 x i64> @test8(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test8:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; CHECK-NEXT:    retq
  %and1 = and <2 x i64> %a, <i64 -1, i64 0>
  %and2 = and <2 x i64> %b, <i64 0, i64 -1>
  %or = or <2 x i64> %and1, %and2
  ret <2 x i64> %or
}


define <4 x i32> @test9(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test9:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %and1 = and <4 x i32> %a, <i32 0, i32 0, i32 -1, i32 -1>
  %and2 = and <4 x i32> %b, <i32 -1, i32 -1, i32 0, i32 0>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}


define <2 x i64> @test10(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test10:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %and1 = and <2 x i64> %a, <i64 0, i64 -1>
  %and2 = and <2 x i64> %b, <i64 -1, i64 0>
  %or = or <2 x i64> %and1, %and2
  ret <2 x i64> %or
}


define <4 x i32> @test11(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test11:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm0[0,1],xmm1[2,3,4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %and1 = and <4 x i32> %a, <i32 -1, i32 0, i32 0, i32 0>
  %and2 = and <4 x i32> %b, <i32 0, i32 -1, i32 -1, i32 -1>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}


define <4 x i32> @test12(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test12:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5,6,7]
; CHECK-NEXT:    retq
  %and1 = and <4 x i32> %a, <i32 0, i32 -1, i32 -1, i32 -1>
  %and2 = and <4 x i32> %b, <i32 -1, i32 0, i32 0, i32 0>
  %or = or <4 x i32> %and1, %and2
  ret <4 x i32> %or
}


; Verify that the following test cases are folded into single shuffles.

define <4 x i32> @test13(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test13:
; CHECK:       # BB#0:
; CHECK-NEXT:    shufps {{.*#+}} xmm0 = xmm0[1,1],xmm1[2,3]
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 1, i32 1, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <2 x i64> @test14(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test14:
; CHECK:       # BB#0:
; CHECK-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}


define <4 x i32> @test15(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test15:
; CHECK:       # BB#0:
; CHECK-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,1],xmm0[2,1]
; CHECK-NEXT:    movaps %xmm1, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 1>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 2, i32 1, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <2 x i64> @test16(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test16:
; CHECK:       # BB#0:
; CHECK-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}


; Verify that the dag-combiner does not fold a OR of two shuffles into a single
; shuffle instruction when the shuffle indexes are not compatible.

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test17:
; CHECK:       # BB#0:
; CHECK-NEXT:    xorps %xmm2, %xmm2
; CHECK-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,1],xmm2[0,0]
; CHECK-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,0],xmm0[0,2]
; CHECK-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2,1,3]
; CHECK-NEXT:    orps %xmm1, %xmm2
; CHECK-NEXT:    movaps %xmm2, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 0, i32 4, i32 2>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <4 x i32> @test18(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test18:
; CHECK:       # BB#0:
; CHECK-NEXT:    xorps %xmm2, %xmm2
; CHECK-NEXT:    xorps %xmm3, %xmm3
; CHECK-NEXT:    blendps {{.*#+}} xmm3 = xmm0[0],xmm3[1,2,3]
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm3[1,0,1,1]
; CHECK-NEXT:    blendps {{.*#+}} xmm2 = xmm1[0],xmm2[1,2,3]
; CHECK-NEXT:    orps %xmm0, %xmm2
; CHECK-NEXT:    movaps %xmm2, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 0, i32 4, i32 4>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 4, i32 4>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <4 x i32> @test19(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test19:
; CHECK:       # BB#0:
; CHECK-NEXT:    xorps %xmm2, %xmm2
; CHECK-NEXT:    xorps %xmm3, %xmm3
; CHECK-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,0],xmm0[0,3]
; CHECK-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,2,1,3]
; CHECK-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,0],xmm1[0,0]
; CHECK-NEXT:    shufps {{.*#+}} xmm2 = xmm2[2,0],xmm1[2,2]
; CHECK-NEXT:    orps %xmm3, %xmm2
; CHECK-NEXT:    movaps %xmm2, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, <4 x i32><i32 4, i32 0, i32 4, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> zeroinitializer, <4 x i32><i32 0, i32 4, i32 2, i32 2>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}


define <2 x i64> @test20(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test20:
; CHECK:       # BB#0:
; CHECK-NEXT:    orps %xmm1, %xmm0
; CHECK-NEXT:    movq %xmm0, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 0, i32 2>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}


define <2 x i64> @test21(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test21:
; CHECK:       # BB#0:
; CHECK-NEXT:    orps %xmm1, %xmm0
; CHECK-NEXT:    movq %xmm0, %xmm0
; CHECK-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <2 x i64> %a, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %shuf2 = shufflevector <2 x i64> %b, <2 x i64> zeroinitializer, <2 x i32><i32 2, i32 0>
  %or = or <2 x i64> %shuf1, %shuf2
  ret <2 x i64> %or
}

; Verify that the DAGCombiner doesn't crash in the attempt to check if a shuffle
; with illegal type has a legal mask. Method 'isShuffleMaskLegal' only knows how to
; handle legal vector value types.
define <4 x i8> @test_crash(<4 x i8> %a, <4 x i8> %b) {
; CHECK-LABEL: test_crash:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    movdqa %xmm1, %xmm0
; CHECK-NEXT:    retq
  %shuf1 = shufflevector <4 x i8> %a, <4 x i8> zeroinitializer, <4 x i32><i32 4, i32 4, i32 2, i32 3>
  %shuf2 = shufflevector <4 x i8> %b, <4 x i8> zeroinitializer, <4 x i32><i32 0, i32 1, i32 4, i32 4>
  %or = or <4 x i8> %shuf1, %shuf2
  ret <4 x i8> %or
}

