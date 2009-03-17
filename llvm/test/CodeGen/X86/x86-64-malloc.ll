; RUN: llvm-as < %s | llc -march=x86-64 | grep {shll.*3, %edi}
; PR3829
; The generated code should multiply by 3 (sizeof i8*) as an i32,
; not as an i64!

define i8** @test(i32 %sz) {
	%sub = add i32 %sz, 536870911		; <i32> [#uses=1]
	%call = malloc i8*, i32 %sub		; <i8**> [#uses=1]
	ret i8** %call
}
