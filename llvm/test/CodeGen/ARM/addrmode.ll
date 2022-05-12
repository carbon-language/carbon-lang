; REQUIRES: asserts
; RUN: llc -mtriple=arm-eabi -stats %s -o - 2>&1 | FileCheck %s

define i32 @t1(i32 %a) {
	%b = mul i32 %a, 9
        %c = inttoptr i32 %b to i32*
        %d = load i32, i32* %c
	ret i32 %d
}

define i32 @t2(i32 %a) {
	%b = mul i32 %a, -7
        %c = inttoptr i32 %b to i32*
        %d = load i32, i32* %c
	ret i32 %d
}

; CHECK: 4 asm-printer

