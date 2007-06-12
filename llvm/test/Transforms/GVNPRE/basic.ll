; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | not grep {%t3 =}
; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | not grep {%t9 =}

define i32 @main() {
block1:
	%t1 = bitcast i32 0 to i32		; <i32> [#uses=5]
	br label %block2

block2:		; preds = %block6, %block1
	%t2 = phi i32 [ %t1, %block1 ], [ %t3, %block6 ]		; <i32> [#uses=3]
	%t3 = add i32 %t2, 1		; <i32> [#uses=5]
	br i1 false, label %block3, label %block7

block3:		; preds = %block2
	br i1 true, label %block4, label %block5

block4:		; preds = %block3
	%t4 = add i32 %t2, %t3		; <i32> [#uses=1]
	%t5 = bitcast i32 %t4 to i32		; <i32> [#uses=1]
	%t6 = add i32 %t1, %t5		; <i32> [#uses=0]
	br label %block6

block5:		; preds = %block3
	%t7 = add i32 %t3, 1		; <i32> [#uses=1]
	br label %block6

block6:		; preds = %block5, %block4
	%t8 = phi i32 [ %t1, %block4 ], [ %t7, %block5 ]		; <i32> [#uses=1]
	%t9 = add i32 %t2, %t3		; <i32> [#uses=2]
	%t10 = add i32 %t9, %t8		; <i32> [#uses=0]
	%t11 = bitcast i32 12 to i32		; <i32> [#uses=1]
	%t12 = add i32 %t9, %t11		; <i32> [#uses=1]
	%t13 = add i32 %t12, %t3		; <i32> [#uses=0]
	br label %block2

block7:		; preds = %block2
	ret i32 %t1
}
