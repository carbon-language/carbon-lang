; RUN: llvm-as < %s | llc -enable-eh
; RUN: llvm-as < %s | llc -enable-eh -march=x86-64 

; PR1326

@__gnat_others_value = external constant i32		; <i32*> [#uses=1]

define void @_ada_eh() {
entry:
	%eh_select = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector( i8* null, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_others_value )		; <i32> [#uses=0]
	ret void
}

declare i32 @llvm.eh.selector(i8*, i8*, ...)

declare i32 @__gnat_eh_personality(...)
