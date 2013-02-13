; RUN: opt < %s -scalarrepl -S | FileCheck %s

define i32 @voltest(i32 %T) {
	%A = alloca {i32, i32}
	%B = getelementptr {i32,i32}* %A, i32 0, i32 0
	store volatile i32 %T, i32* %B
; CHECK: store volatile

	%C = getelementptr {i32,i32}* %A, i32 0, i32 1
	%X = load volatile i32* %C
; CHECK: load volatile
	ret i32 %X
}
