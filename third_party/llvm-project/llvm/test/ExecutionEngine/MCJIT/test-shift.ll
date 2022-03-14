; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define i32 @main() {
	%shamt = add i8 0, 1		; <i8> [#uses=8]
	%shift.upgrd.1 = zext i8 %shamt to i32		; <i32> [#uses=1]
	%t1.s = shl i32 1, %shift.upgrd.1		; <i32> [#uses=0]
	%t2.s = shl i32 1, 4		; <i32> [#uses=0]
	%shift.upgrd.2 = zext i8 %shamt to i32		; <i32> [#uses=1]
	%t1 = shl i32 1, %shift.upgrd.2		; <i32> [#uses=0]
	%t2 = shl i32 1, 5		; <i32> [#uses=0]
	%t2.s.upgrd.3 = shl i64 1, 4		; <i64> [#uses=0]
	%t2.upgrd.4 = shl i64 1, 5		; <i64> [#uses=0]
	%shift.upgrd.5 = zext i8 %shamt to i32		; <i32> [#uses=1]
	%tr1.s = ashr i32 1, %shift.upgrd.5		; <i32> [#uses=0]
	%tr2.s = ashr i32 1, 4		; <i32> [#uses=0]
	%shift.upgrd.6 = zext i8 %shamt to i32		; <i32> [#uses=1]
	%tr1 = lshr i32 1, %shift.upgrd.6		; <i32> [#uses=0]
	%tr2 = lshr i32 1, 5		; <i32> [#uses=0]
	%tr1.l = ashr i64 1, 4		; <i64> [#uses=0]
	%shift.upgrd.7 = zext i8 %shamt to i64		; <i64> [#uses=1]
	%tr2.l = ashr i64 1, %shift.upgrd.7		; <i64> [#uses=0]
	%tr3.l = shl i64 1, 4		; <i64> [#uses=0]
	%shift.upgrd.8 = zext i8 %shamt to i64		; <i64> [#uses=1]
	%tr4.l = shl i64 1, %shift.upgrd.8		; <i64> [#uses=0]
	%tr1.u = lshr i64 1, 5		; <i64> [#uses=0]
	%shift.upgrd.9 = zext i8 %shamt to i64		; <i64> [#uses=1]
	%tr2.u = lshr i64 1, %shift.upgrd.9		; <i64> [#uses=0]
	%tr3.u = shl i64 1, 5		; <i64> [#uses=0]
	%shift.upgrd.10 = zext i8 %shamt to i64		; <i64> [#uses=1]
	%tr4.u = shl i64 1, %shift.upgrd.10		; <i64> [#uses=0]
	ret i32 0
}
