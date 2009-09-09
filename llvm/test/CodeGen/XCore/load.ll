; RUN: llc < %s -march=xcore > %t1.s
; RUN: not grep add %t1.s
; RUN: not grep ldaw %t1.s
; RUN: not grep lda16 %t1.s
; RUN: not grep zext %t1.s
; RUN: not grep sext %t1.s
; RUN: grep "ldw" %t1.s | count 2
; RUN: grep "ld16s" %t1.s | count 1
; RUN: grep "ld8u" %t1.s | count 1

define i32 @load32(i32* %p, i32 %offset) nounwind {
entry:
	%0 = getelementptr i32* %p, i32 %offset
	%1 = load i32* %0, align 4
	ret i32 %1
}

define i32 @load32_imm(i32* %p) nounwind {
entry:
	%0 = getelementptr i32* %p, i32 11
	%1 = load i32* %0, align 4
	ret i32 %1
}

define i32 @load16(i16* %p, i32 %offset) nounwind {
entry:
	%0 = getelementptr i16* %p, i32 %offset
	%1 = load i16* %0, align 2
	%2 = sext i16 %1 to i32
	ret i32 %2
}

define i32 @load8(i8* %p, i32 %offset) nounwind {
entry:
	%0 = getelementptr i8* %p, i32 %offset
	%1 = load i8* %0, align 1
	%2 = zext i8 %1 to i32
	ret i32 %2
}
