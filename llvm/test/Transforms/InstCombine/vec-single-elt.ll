; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep element
; insert/extractelement should canonicalize to bitcast

define i64 @a(<1 x i64> %x) {
  %r = extractelement <1 x i64> %x, i32 0
  ret i64 %r
}

define <1 x i64> @b(i64 %x) {
  %r = insertelement <1 x i64> undef, i64 %x, i32 0
  ret <1 x i64> %r
}
