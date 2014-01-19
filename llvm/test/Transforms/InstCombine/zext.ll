; Tests to make sure elimination of casts is working correctly
; RUN: opt < %s -instcombine -S | FileCheck %s

define i64 @test_sext_zext(i16 %A) {
        %c1 = zext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = sext i32 %c1 to i64               ; <i64> [#uses=1]
        ret i64 %c2

; CHECK-LABEL: @test_sext_zext
; CHECK-NOT: %c1
; CHECK: %c2 = zext i16 %A to i64
; CHECK: ret i64 %c2
}

define <2 x i64> @test2(<2 x i1> %A) {
  %xor = xor <2 x i1> %A, <i1 true, i1 true>
  %zext = zext <2 x i1> %xor to <2 x i64>
  ret <2 x i64> %zext

; CHECK-LABEL: @test2
; CHECK-NEXT: zext <2 x i1> %A to <2 x i64>
; CHECK-NEXT: xor <2 x i64> %1, <i64 1, i64 1>
}

define <2 x i64> @test3(<2 x i64> %A) {
  %trunc = trunc <2 x i64> %A to <2 x i32>
  %and = and <2 x i32> %trunc, <i32 23, i32 42>
  %zext = zext <2 x i32> %and to <2 x i64>
  ret <2 x i64> %zext

; CHECK-LABEL: @test3
; CHECK-NEXT: and <2 x i64> %A, <i64 23, i64 42>
}

define <2 x i64> @test4(<2 x i64> %A) {
  %trunc = trunc <2 x i64> %A to <2 x i32>
  %and = and <2 x i32> %trunc, <i32 23, i32 42>
  %xor = xor <2 x i32> %and, <i32 23, i32 42>
  %zext = zext <2 x i32> %xor to <2 x i64>
  ret <2 x i64> %zext

; CHECK-LABEL: @test4
; CHECK-NEXT: xor <2 x i64> %A, <i64 4294967295, i64 4294967295>
; CHECK-NEXT: and <2 x i64> %1, <i64 23, i64 42>
}
