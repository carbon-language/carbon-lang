; RUN: llvm-as < %s | llc -march=c | not grep "\-\-65535"
; ModuleID = '<stdin>'
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

declare void %func(int)

void %funcb() {
entry:
	%tmp.1 = sub int 0, -65535		; <int> [#uses=1]
	call void %func( int %tmp.1 )
	br label %return

return:		; preds = %entry
	ret void
}
