; RUN: opt < %s -instcombine -S | not grep bitcast

define <4 x i32> @a(<1 x i64> %y) {
  %c = bitcast <2 x i64> <i64 0, i64 0> to <4 x i32>
  ret <4 x i32> %c
}

define <4 x i32> @b(<1 x i64> %y) {
  %c = bitcast <2 x i64> <i64 -1, i64 -1> to <4 x i32>
  ret <4 x i32> %c
}



