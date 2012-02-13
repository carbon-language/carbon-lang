; RUN: llc < %s -march=x86-64
; PR 9267

define<4 x i64> @func_32_64() {
  %F = load <4 x i32>* undef
  %G = zext <4 x i32> %F to <4 x i64>
  %H = load <4 x i32>* undef
  %Y = zext <4 x i32> %H to <4 x i64>
  %T = or <4 x i64> %Y, %G
  ret <4 x i64> %T
}
