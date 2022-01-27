; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; PR2589

define void @main({ i32 }*) {
entry:
	%sret1 = alloca { i32 }		; <{ i32 }*> [#uses=1]
	load { i32 }, { i32 }* %sret1		; <{ i32 }>:1 [#uses=0]
	ret void
}
