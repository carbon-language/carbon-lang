; RUN: llc < %s -march=systemz | grep sra   | count 6
; RUN: llc < %s -march=systemz | grep srag  | count 3
; RUN: llc < %s -march=systemz | grep srl   | count 6
; RUN: llc < %s -march=systemz | grep srlg  | count 3
; RUN: llc < %s -march=systemz | grep sll   | count 6
; RUN: llc < %s -march=systemz | grep sllg  | count 3

define signext i32 @foo1(i32 %a, i32 %idx) nounwind readnone {
entry:
	%add = add i32 %idx, 1		; <i32> [#uses=1]
	%shr = ashr i32 %a, %add		; <i32> [#uses=1]
	ret i32 %shr
}

define signext i32 @foo2(i32 %a, i32 %idx) nounwind readnone {
entry:
	%add = add i32 %idx, 1		; <i32> [#uses=1]
	%shr = shl i32 %a, %add		; <i32> [#uses=1]
	ret i32 %shr
}

define signext i32 @foo3(i32 %a, i32 %idx) nounwind readnone {
entry:
	%add = add i32 %idx, 1		; <i32> [#uses=1]
	%shr = lshr i32 %a, %add		; <i32> [#uses=1]
	ret i32 %shr
}

define signext i64 @foo4(i64 %a, i64 %idx) nounwind readnone {
entry:
	%add = add i64 %idx, 1		; <i64> [#uses=1]
	%shr = ashr i64 %a, %add		; <i64> [#uses=1]
	ret i64 %shr
}

define signext i64 @foo5(i64 %a, i64 %idx) nounwind readnone {
entry:
	%add = add i64 %idx, 1		; <i64> [#uses=1]
	%shr = shl i64 %a, %add		; <i64> [#uses=1]
	ret i64 %shr
}

define signext i64 @foo6(i64 %a, i64 %idx) nounwind readnone {
entry:
	%add = add i64 %idx, 1		; <i64> [#uses=1]
	%shr = lshr i64 %a, %add		; <i64> [#uses=1]
	ret i64 %shr
}

define signext i32 @foo7(i32 %a, i32 %idx) nounwind readnone {
entry:
        %shr = ashr i32 %a, 1
        ret i32 %shr
}

define signext i32 @foo8(i32 %a, i32 %idx) nounwind readnone {
entry:
        %shr = shl i32 %a, 1
        ret i32 %shr
}

define signext i32 @foo9(i32 %a, i32 %idx) nounwind readnone {
entry:
        %shr = lshr i32 %a, 1
        ret i32 %shr
}

define signext i32 @foo10(i32 %a, i32 %idx) nounwind readnone {
entry:
        %shr = ashr i32 %a, %idx
        ret i32 %shr
}

define signext i32 @foo11(i32 %a, i32 %idx) nounwind readnone {
entry:
        %shr = shl i32 %a, %idx
        ret i32 %shr
}

define signext i32 @foo12(i32 %a, i32 %idx) nounwind readnone {
entry:
        %shr = lshr i32 %a, %idx
        ret i32 %shr
}

define signext i64 @foo13(i64 %a, i64 %idx) nounwind readnone {
entry:
        %shr = ashr i64 %a, 1
        ret i64 %shr
}

define signext i64 @foo14(i64 %a, i64 %idx) nounwind readnone {
entry:
        %shr = shl i64 %a, 1
        ret i64 %shr
}

define signext i64 @foo15(i64 %a, i64 %idx) nounwind readnone {
entry:
        %shr = lshr i64 %a, 1
        ret i64 %shr
}

define signext i64 @foo16(i64 %a, i64 %idx) nounwind readnone {
entry:
        %shr = ashr i64 %a, %idx
        ret i64 %shr
}

define signext i64 @foo17(i64 %a, i64 %idx) nounwind readnone {
entry:
        %shr = shl i64 %a, %idx
        ret i64 %shr
}

define signext i64 @foo18(i64 %a, i64 %idx) nounwind readnone {
entry:
        %shr = lshr i64 %a, %idx
        ret i64 %shr
}

