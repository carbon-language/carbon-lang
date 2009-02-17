; Tests to make sure elimination of casts is working correctly
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   notcast {} {%c1.*}

define i64 @test_sext_zext(i16 %A) {
        %c1 = zext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = sext i32 %c1 to i64               ; <i64> [#uses=1]
        ret i64 %c2
}

; PR3599
define i32 @test2(i64 %tmp) nounwind readnone {
entry:
	%tmp5 = trunc i64 %tmp to i8		; <i8> [#uses=1]
	%tmp7 = lshr i64 %tmp, 8		; <i64> [#uses=1]
	%tmp8 = trunc i64 %tmp7 to i8		; <i8> [#uses=1]
	%tmp10 = lshr i64 %tmp, 16		; <i64> [#uses=1]
	%tmp11 = trunc i64 %tmp10 to i8		; <i8> [#uses=1]
	%tmp13 = lshr i64 %tmp, 24		; <i64> [#uses=1]
	%tmp14 = trunc i64 %tmp13 to i8		; <i8> [#uses=1]
	%tmp1 = zext i8 %tmp5 to i32		; <i32> [#uses=1]
	%tmp2 = zext i8 %tmp8 to i32		; <i32> [#uses=1]
	%tmp3 = shl i32 %tmp2, 8		; <i32> [#uses=1]
	%tmp4 = zext i8 %tmp11 to i32		; <i32> [#uses=1]
	%tmp6 = shl i32 %tmp4, 16		; <i32> [#uses=1]
	%tmp9 = zext i8 %tmp14 to i32		; <i32> [#uses=1]
	%tmp12 = shl i32 %tmp9, 24		; <i32> [#uses=1]
	%tmp15 = or i32 %tmp12, %tmp1		; <i32> [#uses=1]
	%tmp16 = or i32 %tmp15, %tmp6		; <i32> [#uses=1]
	%tmp17 = or i32 %tmp16, %tmp3		; <i32> [#uses=1]
	ret i32 %tmp17
}

