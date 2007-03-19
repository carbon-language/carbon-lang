; Basic test for bugpoint.
; RUN: bugpoint %s -domset -idom -domset -bugpoint-crashcalls \
; RUN:   -domset -idom -domset

define i32 @test() {
	call i32 @test()
	ret i32 %1
}
