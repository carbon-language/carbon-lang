; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep shl | count 1
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep ashr | count 1
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep lshr | count 1

define <4 x i32> @foo(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
	%cmp = shl <4 x i32> %a, %b		; <4 x i32> [#uses=1]
	ret <4 x i32> %cmp
}

define <4 x i32> @bar(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
	%cmp = lshr <4 x i32> %a, %b		; <4 x i32> [#uses=1]
	ret <4 x i32> %cmp
}

define <4 x i32> @baz(<4 x i32> %a, <4 x i32> %b) nounwind  {
entry:
	%cmp = ashr <4 x i32> %a, %b		; <4 x i32> [#uses=1]
	ret <4 x i32> %cmp
}

; Constant expressions: these should be folded.
define <2 x i64> @foo_ce() nounwind {
  ret <2 x i64> shl (<2 x i64> <i64 5, i64 6>, <2 x i64> <i64 3, i64 5>)
}
define <2 x i64> @bar_ce() nounwind {
  ret <2 x i64> lshr (<2 x i64> <i64 340, i64 380>, <2 x i64> <i64 3, i64 5>)
}
define <2 x i64> @baz_ce() nounwind {
  ret <2 x i64> ashr (<2 x i64> <i64 573, i64 411>, <2 x i64> <i64 3, i64 5>)
}
