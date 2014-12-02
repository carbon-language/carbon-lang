; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 < %s | FileCheck %s
;
; Verify that the DAGCombiner is able to fold a vector AND into a blend
; if one of the operands to the AND is a vector of all constants, and each
; constant element is either zero or all-ones.


define <4 x i32> @test1(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 0, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test1
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3,4,5,6,7]
; CHECK-NEXT: retq


define <4 x i32> @test2(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 0, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test2
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5,6,7]
; CHECK-NEXT: retq


define <4 x i32> @test3(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 0, i32 -1, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test3
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5],xmm1[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test4(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 0, i32 0, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test4
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5],xmm0[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test5(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test5
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5],xmm1[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test6(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 0, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test6
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test7(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 0, i32 -1, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test7
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT: retq


define <4 x i32> @test8(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 0, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test8
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3,4,5],xmm0[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test9(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 -1, i32 0, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test9
; CHECK: movq %xmm0, %xmm0
; CHECK-NEXT: retq


define <4 x i32> @test10(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 -1, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test10
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5],xmm1[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test11(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test11
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5,6,7]
; CHECK-NEXT: retq


define <4 x i32> @test12(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 -1, i32 -1, i32 0>
  ret <4 x i32> %1
}
; CHECK-LABEL: test12
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test13(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 -1, i32 0, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test13
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test14(<4 x i32> %A) {
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 -1>
  ret <4 x i32> %1
}
; CHECK-LABEL: test14
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; CHECK-NEXT: retq


define <4 x i32> @test15(<4 x i32> %A, <4 x i32> %B) {
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 -1>
  %2 = and <4 x i32> %B, <i32 0, i32 -1, i32 0, i32 0>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}
; CHECK-LABEL: test15
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; CHECK-NEXT: retq


define <4 x i32> @test16(<4 x i32> %A, <4 x i32> %B) {
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 0>
  %2 = and <4 x i32> %B, <i32 0, i32 -1, i32 0, i32 -1>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}
; CHECK-LABEL: test16
; CHECK: pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5],xmm1[6,7]
; CHECK-NEXT: retq


define <4 x i32> @test17(<4 x i32> %A, <4 x i32> %B) {
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 0, i32 -1>
  %2 = and <4 x i32> %B, <i32 -1, i32 0, i32 -1, i32 0>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}
; CHECK-LABEL: test17
; CHECK: pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT: retq
