; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @a
; CHECK-NOT: bitcast
; CHECK: ret
define <4 x i32> @a(<1 x i64> %y) {
  %c = bitcast <2 x i64> <i64 0, i64 0> to <4 x i32>
  ret <4 x i32> %c
}

; CHECK: @b
; CHECK: bitcast
; CHECK: ret

define <4 x i32> @b(<1 x i64> %y) {
  %c = bitcast <2 x i64> <i64 -1, i64 -1> to <4 x i32>
  ret <4 x i32> %c
}

; CHECK: @foo
; CHECK: bitcast

; from MultiSource/Benchmarks/Bullet
define <2 x float> @foo() {
  %cast = bitcast i64 -1 to <2 x float>
  ret <2 x float> %cast
}



