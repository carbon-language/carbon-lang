; RUN: opt < %s -instcombine -S | FileCheck %s

define i64 @test1(i64 %A, i32 %B) {
        %tmp12 = zext i32 %B to i64
        %tmp3 = shl i64 %tmp12, 32
        %tmp5 = add i64 %tmp3, %A
        %tmp6 = and i64 %tmp5, 123
        ret i64 %tmp6
; CHECK-LABEL: @test1(
; CHECK-NEXT: and i64 %A, 123
; CHECK-NEXT: ret i64
}

define i32 @test2(i32 %A) {
  %B = and i32 %A, 7
  %C = and i32 %A, 32
  %F = add i32 %B, %C
  ret i32 %F
; CHECK-LABEL: @test2(
; CHECK-NEXT: and i32 %A, 39
; CHECK-NEXT: ret i32
}

define i32 @test3(i32 %A) {
  %B = and i32 %A, 128
  %C = lshr i32 %A, 30
  %F = add i32 %B, %C
  ret i32 %F
; CHECK-LABEL: @test3(
; CHECK-NEXT: and
; CHECK-NEXT: lshr
; CHECK-NEXT: or i32 %B, %C
; CHECK-NEXT: ret i32
}

define i32 @test4(i32 %A) {
  %B = add nuw i32 %A, %A
  ret i32 %B
; CHECK-LABEL: @test4(
; CHECK-NEXT: %B = shl nuw i32 %A, 1
; CHECK-NEXT: ret i32 %B
}

define <2 x i1> @test5(<2 x i1> %A, <2 x i1> %B) {
  %add = add <2 x i1> %A, %B
  ret <2 x i1> %add
; CHECK-LABEL: @test5(
; CHECK-NEXT: %add = xor <2 x i1> %A, %B
; CHECK-NEXT: ret <2 x i1> %add
}

define <2 x i64> @test6(<2 x i64> %A) {
  %shl = shl <2 x i64> %A, <i64 2, i64 3>
  %add = add <2 x i64> %shl, %A
  ret <2 x i64> %add
; CHECK-LABEL: @test6(
; CHECK-NEXT: %add = mul <2 x i64> %A, <i64 5, i64 9>
; CHECK-NEXT: ret <2 x i64> %add
}

define <2 x i64> @test7(<2 x i64> %A) {
  %shl = shl <2 x i64> %A, <i64 2, i64 3>
  %mul = mul <2 x i64> %A, <i64 3, i64 4>
  %add = add <2 x i64> %shl, %mul
  ret <2 x i64> %add
; CHECK-LABEL: @test7(
; CHECK-NEXT: %add = mul <2 x i64> %A, <i64 7, i64 12>
; CHECK-NEXT: ret <2 x i64> %add
}

define <2 x i64> @test8(<2 x i64> %A) {
  %xor = xor <2 x i64> %A, <i64 -1, i64 -1>
  %add = add <2 x i64> %xor, <i64 2, i64 3>
  ret <2 x i64> %add
; CHECK-LABEL: @test8(
; CHECK-NEXT: %add = sub <2 x i64> <i64 1, i64 2>, %A
; CHECK-NEXT: ret <2 x i64> %add
}
