; RUN: opt < %s -functionattrs -S | not grep read
; PR3754

define i8* @m(i32 %size) {
	%tmp = malloc i8, i32 %size		; <i8*> [#uses=1]
	ret i8* %tmp
}
