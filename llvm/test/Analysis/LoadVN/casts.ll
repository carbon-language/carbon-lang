; Check to make sure that Value Numbering doesn't merge casts of different
; flavors.
; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | \
; RUN:   grep {\[sz\]ext} | count 2

declare void @external(i32)

define i32 @test_casts(i16 %x) {
	%a = sext i16 %x to i32		; <i32> [#uses=1]
	%b = zext i16 %x to i32		; <i32> [#uses=1]
	call void @external( i32 %a )
	ret i32 %b
}
