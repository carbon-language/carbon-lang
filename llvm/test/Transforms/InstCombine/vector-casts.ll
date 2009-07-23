; RUN: llvm-as < %s | opt -instcombine > %t
; RUN: not grep trunc %t
; RUN: not grep ashr %t

; This turns into a&1 != 0
define <2 x i1> @a(<2 x i64> %a) {
  %t = trunc <2 x i64> %a to <2 x i1>
  ret <2 x i1> %t
}
; The ashr turns into an lshr.
define <2 x i64> @b(<2 x i64> %a) {
  %b = and <2 x i64> %a, <i64 65535, i64 65535>
  %t = ashr <2 x i64> %b, <i64 1, i64 1>
  ret <2 x i64> %t
}
