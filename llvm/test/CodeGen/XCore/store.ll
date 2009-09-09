; RUN: llc < %s -march=xcore > %t1.s
; RUN: not grep add %t1.s
; RUN: not grep ldaw %t1.s
; RUN: not grep lda16 %t1.s
; RUN: grep "stw" %t1.s | count 2
; RUN: grep "st16" %t1.s | count 1
; RUN: grep "st8" %t1.s | count 1

define void @store32(i32* %p, i32 %offset, i32 %val) nounwind {
entry:
	%0 = getelementptr i32* %p, i32 %offset
	store i32 %val, i32* %0, align 4
	ret void
}

define void @store32_imm(i32* %p, i32 %val) nounwind {
entry:
	%0 = getelementptr i32* %p, i32 11
	store i32 %val, i32* %0, align 4
	ret void
}

define void @store16(i16* %p, i32 %offset, i16 %val) nounwind {
entry:
	%0 = getelementptr i16* %p, i32 %offset
	store i16 %val, i16* %0, align 2
	ret void
}

define void @store8(i8* %p, i32 %offset, i8 %val) nounwind {
entry:
	%0 = getelementptr i8* %p, i32 %offset
	store i8 %val, i8* %0, align 1
	ret void
}
