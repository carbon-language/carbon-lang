; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | grep nounwind | count 2
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | grep signext | count 2
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep inreg
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep zeroext
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep byval

	%struct = type { }

@g = global i8 0

define internal i8 @foo(i8* inreg %p, i8 signext %y, ... ) zeroext nounwind {
	store i8 %y, i8* @g
	ret i8 0
}

define i32 @bar() {
	%A = call i8(i8*, i8, ...)* @foo(i8* inreg null, i8 signext 1, %struct* byval null ) zeroext nounwind
	ret i32 0
}
