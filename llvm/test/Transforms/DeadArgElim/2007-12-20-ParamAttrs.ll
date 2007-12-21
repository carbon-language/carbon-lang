; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | grep nounwind | count 2
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | grep signext | count 2
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep inreg
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep zeroext

@g = global i8 0

define internal i8 @foo(i8* inreg %p, i8 signext %y, ... ) zeroext nounwind {
	store i8 %y, i8* @g
	ret i8 0
}

define i32 @bar() {
	%A = call i8(i8*, i8, ...)* @foo(i8* inreg null, i8 signext 1, i8 2) zeroext nounwind
	ret i32 0
}
