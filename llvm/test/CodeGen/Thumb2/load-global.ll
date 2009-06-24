; RUN: llvm-as < %s | llc -mtriple=thumbv7-apple-darwin
; RUN: llvm-as < %s | llc -mtriple=thumbv7-apple-darwin -relocation-model=pic | grep add | grep pc

@G = external global i32

define i32 @test1() {
	%tmp = load i32* @G
	ret i32 %tmp
}
