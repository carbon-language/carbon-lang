; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s

; Verify that the DAGCombiner correctly folds according to the following rules:

; fold (AND (shuf (A, C), shuf (B, C)) -> shuf (AND (A, B), C)
; fold (OR  (shuf (A, C), shuf (B, C)) -> shuf (OR  (A, B), C)
; fold (XOR (shuf (A, C), shuf (B, C)) -> shuf (XOR (A, B), V_0)

; fold (AND (shuf (C, A), shuf (C, B)) -> shuf (C, AND (A, B))
; fold (OR  (shuf (C, A), shuf (C, B)) -> shuf (C, OR  (A, B))
; fold (XOR (shuf (C, A), shuf (C, B)) -> shuf (V_0, XOR (A, B))



define <4 x i32> @test1(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}
; CHECK-LABEL: test1
; CHECK-NOT: pshufd
; CHECK: pand
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @test2(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test2
; CHECK-NOT: pshufd
; CHECK: por
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @test3(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
; CHECK-LABEL: test3
; CHECK-NOT: pshufd
; CHECK: pxor
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @test4(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}
; CHECK-LABEL: test4
; CHECK-NOT: pshufd
; CHECK: pand
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @test5(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test5
; CHECK-NOT: pshufd
; CHECK: por
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


define <4 x i32> @test6(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
; CHECK-LABEL: test6
; CHECK-NOT: pshufd
; CHECK: pxor
; CHECK-NEXT: pshufd
; CHECK-NEXT: ret


; Verify that DAGCombiner moves the shuffle after the xor/and/or even if shuffles
; are not performing a swizzle operations.

define <4 x i32> @test1b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}
; CHECK-LABEL: test1b
; CHECK-NOT: blendps
; CHECK: andps
; CHECK-NEXT: blendps
; CHECK-NEXT: ret


define <4 x i32> @test2b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test2b
; CHECK-NOT: blendps
; CHECK: orps
; CHECK-NEXT: blendps
; CHECK-NEXT: ret


define <4 x i32> @test3b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
; CHECK-LABEL: test3b
; CHECK-NOT: blendps
; CHECK: xorps
; CHECK-NEXT: xorps
; CHECK-NEXT: blendps
; CHECK-NEXT: ret


define <4 x i32> @test4b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}
; CHECK-LABEL: test4b
; CHECK-NOT: blendps
; CHECK: andps
; CHECK-NEXT: blendps
; CHECK: ret


define <4 x i32> @test5b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test5b
; CHECK-NOT: blendps
; CHECK: orps
; CHECK-NEXT: blendps
; CHECK: ret


define <4 x i32> @test6b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
; CHECK-LABEL: test6b
; CHECK-NOT: blendps
; CHECK: xorps
; CHECK-NEXT: xorps
; CHECK-NEXT: blendps
; CHECK: ret

define <4 x i32> @test1c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}
; CHECK-LABEL: test1c
; CHECK-NOT: shufps
; CHECK: andps
; CHECK-NEXT: shufps
; CHECK-NEXT: ret


define <4 x i32> @test2c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test2c
; CHECK-NOT: shufps
; CHECK: orps
; CHECK-NEXT: shufps
; CHECK-NEXT: ret


define <4 x i32> @test3c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
; CHECK-LABEL: test3c
; CHECK-NOT: shufps
; CHECK: xorps
; CHECK-NEXT: xorps
; CHECK-NEXT: shufps
; CHECK-NEXT: ret


define <4 x i32> @test4c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}
; CHECK-LABEL: test4c
; CHECK-NOT: shufps
; CHECK: andps
; CHECK-NEXT: shufps
; CHECK: ret


define <4 x i32> @test5c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}
; CHECK-LABEL: test5c
; CHECK-NOT: shufps
; CHECK: orps
; CHECK-NEXT: shufps
; CHECK: ret


define <4 x i32> @test6c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
; CHECK-LABEL: test6c
; CHECK-NOT: shufps
; CHECK: xorps
; CHECK-NEXT: xorps
; CHECK-NEXT: shufps
; CHECK: ret

