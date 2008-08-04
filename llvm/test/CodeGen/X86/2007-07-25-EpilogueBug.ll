; Verify that the addl comes before any popl.

; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -disable-correct-folding -mcpu=i386 | \
; RUN:   %prcontext ret 1 | grep popl
; PR1573

	%struct.c34006f__TsB = type { i8, i32, i32, %struct.c34006f__TsB___b___XVN }
	%struct.c34006f__TsB___b___XVN = type { %struct.c34006f__TsB___b___XVN___O }
	%struct.c34006f__TsB___b___XVN___O = type { float }

define fastcc i8 @c34006f__pkg__parentEQ.213(%struct.c34006f__TsB* %x, %struct.c34006f__TsB* %y) zeroext  {
entry:
	%tmp190 = icmp eq i8 0, 0		; <i1> [#uses=1]
	%tmp207 = icmp eq i32 0, 0		; <i1> [#uses=1]
	%bothcond = and i1 %tmp190, %tmp207		; <i1> [#uses=1]
	%tmp224 = icmp eq i32 0, 0		; <i1> [#uses=1]
	%bothcond1 = and i1 %bothcond, %tmp224		; <i1> [#uses=1]
	br i1 %bothcond1, label %cond_next229, label %UnifiedReturnBlock

cond_next229:		; preds = %entry
	%tmp234 = icmp eq i8 0, 0		; <i1> [#uses=1]
	br i1 %tmp234, label %cond_false391, label %cond_true237

cond_true237:		; preds = %cond_next229
	%tmp268 = icmp sgt i32 0, -1		; <i1> [#uses=2]
	%max269 = select i1 %tmp268, i32 0, i32 0		; <i32> [#uses=1]
	%tmp305.op = add i32 0, -1		; <i32> [#uses=1]
	br i1 false, label %bb328, label %cond_next315

cond_next315:		; preds = %cond_true237
	ret i8 0

bb328:		; preds = %cond_true237
	%tmp337 = select i1 %tmp268, i32 %tmp305.op, i32 -1		; <i32> [#uses=1]
	%tmp347 = icmp eq i32 %tmp337, 0		; <i1> [#uses=1]
	br i1 %tmp347, label %cond_next351, label %UnifiedReturnBlock

cond_next351:		; preds = %bb328
	%tmp354 = getelementptr %struct.c34006f__TsB* %y, i32 0, i32 3		; <%struct.c34006f__TsB___b___XVN*> [#uses=1]
	%tmp354355 = bitcast %struct.c34006f__TsB___b___XVN* %tmp354 to i8*		; <i8*> [#uses=1]
	%tmp358 = getelementptr %struct.c34006f__TsB* %x, i32 0, i32 3		; <%struct.c34006f__TsB___b___XVN*> [#uses=1]
	%tmp358359 = bitcast %struct.c34006f__TsB___b___XVN* %tmp358 to i8*		; <i8*> [#uses=1]
	%tmp360 = tail call i32 (i8*, i8*, i32, ...)* @memcmp( i8* %tmp358359, i8* %tmp354355, i32 %max269 )		; <i32> [#uses=0]
	ret i8 0

cond_false391:		; preds = %cond_next229
	%tmp1 = getelementptr %struct.c34006f__TsB* %y, i32 0, i32 3, i32 0, i32 0
        %tmp2 = load float* %tmp1
	%tmp400 = fcmp une float %tmp2, 0.000000e+00		; <i1> [#uses=1]
	%not.tmp400 = xor i1 %tmp400, true		; <i1> [#uses=1]
	%retval = zext i1 %not.tmp400 to i8		; <i8> [#uses=1]
	ret i8 %retval

UnifiedReturnBlock:		; preds = %bb328, %entry
	ret i8 0
}

declare i32 @memcmp(i8*, i8*, i32, ...)
