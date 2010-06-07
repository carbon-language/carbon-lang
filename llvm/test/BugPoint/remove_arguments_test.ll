; RUN: bugpoint %s -output-prefix %t -bugpoint-crashcalls -silence-passes
; RUN: llvm-dis remove_arguments_test.ll.tmp-reduced-simplified.bc -o - | FileCheck %s

; Test to make sure that arguments are removed from the function if they are 
; unnecessary.

declare i32 @test2()

; CHECK: define void @test() {
define i32 @test(i32 %A, i32 %B, float %C) {
	call i32 @test2()
	ret i32 %1
}
