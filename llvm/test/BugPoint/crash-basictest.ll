; Basic test for bugpoint.
; RUN: bugpoint %s -idom -bugpoint-crashcalls \
; RUN:   -idom

define i32 @test() {
	call i32 @test()
	ret i32 %1
}
