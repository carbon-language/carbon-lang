; RUN: llvm-as < %s | llc -march=x86-64 > %t
; RUN: grep negl %t | count 1
; RUN: not grep {sub\[bwlq\]} %t
; RUN: grep mov %t | count 1
; RUN: grep {leal	-4(} %t | count 1

; ISel the add of -4 with a neg and use an lea for the rest of the
; arithemtic.

define i32 @test(i32 %x_offs) nounwind readnone {
entry:
	%t0 = icmp sgt i32 %x_offs, 4		; <i1> [#uses=1]
	br i1 %t0, label %bb.nph, label %bb2

bb.nph:		; preds = %entry
	%tmp = add i32 %x_offs, -5		; <i32> [#uses=1]
	%tmp6 = lshr i32 %tmp, 2		; <i32> [#uses=1]
	%tmp7 = mul i32 %tmp6, -4		; <i32> [#uses=1]
	%tmp8 = add i32 %tmp7, %x_offs		; <i32> [#uses=1]
	%tmp9 = add i32 %tmp8, -4		; <i32> [#uses=1]
	ret i32 %tmp9

bb2:		; preds = %entry
	ret i32 %x_offs
}
