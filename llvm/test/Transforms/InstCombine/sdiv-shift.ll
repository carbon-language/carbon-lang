; RUN: opt < %s -instcombine -S | not grep div

define i32 @a(i16 zeroext %x, i32 %y) nounwind {
entry:
	%conv = zext i16 %x to i32
	%s = shl i32 2, %y
	%d = sdiv i32 %conv, %s
	ret i32 %d
}
