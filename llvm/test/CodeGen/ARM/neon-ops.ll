; RUN: llc -march=arm -mattr=+neon -O2 -o /dev/null

; This used to crash.
define <4 x i32> @test1(<4 x i16> %a) {
  %A = zext <4 x i16> %a to <4 x i32>
  ret <4 x i32> %A
}
