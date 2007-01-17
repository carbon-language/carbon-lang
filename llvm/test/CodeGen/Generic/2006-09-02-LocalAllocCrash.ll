; RUN: llvm-upgrade < %s | llvm-as | llc -regalloc=local

	%struct.CHESS_POSITION = type { ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, uint, int, sbyte, sbyte, [64 x sbyte], sbyte, sbyte, sbyte, sbyte, sbyte }
%search = external global %struct.CHESS_POSITION		; <%struct.CHESS_POSITION*> [#uses=2]
%bishop_shift_rl45 = external global [64 x int]		; <[64 x int]*> [#uses=1]
%bishop_shift_rr45 = external global [64 x int]		; <[64 x int]*> [#uses=1]
%black_outpost = external global [64 x sbyte]		; <[64 x sbyte]*> [#uses=1]
%bishop_mobility_rl45 = external global [64 x [256 x int]]		; <[64 x [256 x int]]*> [#uses=1]
%bishop_mobility_rr45 = external global [64 x [256 x int]]		; <[64 x [256 x int]]*> [#uses=1]

implementation   ; Functions:

declare fastcc int %FirstOne()

fastcc void %Evaluate() {
entry:
	br bool false, label %cond_false186, label %cond_true

cond_true:		; preds = %entry
	ret void

cond_false186:		; preds = %entry
	br bool false, label %cond_true293, label %bb203

bb203:		; preds = %cond_false186
	ret void

cond_true293:		; preds = %cond_false186
	br bool false, label %cond_true298, label %cond_next317

cond_true298:		; preds = %cond_true293
	br bool false, label %cond_next518, label %cond_true397.preheader

cond_next317:		; preds = %cond_true293
	ret void

cond_true397.preheader:		; preds = %cond_true298
	ret void

cond_next518:		; preds = %cond_true298
	br bool false, label %bb1069, label %cond_true522

cond_true522:		; preds = %cond_next518
	ret void

bb1069:		; preds = %cond_next518
	br bool false, label %cond_next1131, label %bb1096

bb1096:		; preds = %bb1069
	ret void

cond_next1131:		; preds = %bb1069
	br bool false, label %cond_next1207, label %cond_true1150

cond_true1150:		; preds = %cond_next1131
	ret void

cond_next1207:		; preds = %cond_next1131
	br bool false, label %cond_next1219, label %cond_true1211

cond_true1211:		; preds = %cond_next1207
	ret void

cond_next1219:		; preds = %cond_next1207
	br bool false, label %cond_true1223, label %cond_next1283

cond_true1223:		; preds = %cond_next1219
	br bool false, label %cond_true1254, label %cond_true1264

cond_true1254:		; preds = %cond_true1223
	br bool false, label %bb1567, label %cond_true1369.preheader

cond_true1264:		; preds = %cond_true1223
	ret void

cond_next1283:		; preds = %cond_next1219
	ret void

cond_true1369.preheader:		; preds = %cond_true1254
	ret void

bb1567:		; preds = %cond_true1254
	%tmp1580 = load ulong* getelementptr (%struct.CHESS_POSITION* %search, int 0, uint 3)		; <ulong> [#uses=1]
	%tmp1591 = load ulong* getelementptr (%struct.CHESS_POSITION* %search, int 0, uint 4)		; <ulong> [#uses=1]
	%tmp1572 = tail call fastcc int %FirstOne( )		; <int> [#uses=5]
	%tmp1582 = getelementptr [64 x int]* %bishop_shift_rl45, int 0, int %tmp1572		; <int*> [#uses=1]
	%tmp1583 = load int* %tmp1582		; <int> [#uses=1]
	%tmp1583 = cast int %tmp1583 to ubyte		; <ubyte> [#uses=1]
	%tmp1584 = shr ulong %tmp1580, ubyte %tmp1583		; <ulong> [#uses=1]
	%tmp1584 = cast ulong %tmp1584 to uint		; <uint> [#uses=1]
	%tmp1585 = and uint %tmp1584, 255		; <uint> [#uses=1]
	%tmp1587 = getelementptr [64 x [256 x int]]* %bishop_mobility_rl45, int 0, int %tmp1572, uint %tmp1585		; <int*> [#uses=1]
	%tmp1588 = load int* %tmp1587		; <int> [#uses=1]
	%tmp1593 = getelementptr [64 x int]* %bishop_shift_rr45, int 0, int %tmp1572		; <int*> [#uses=1]
	%tmp1594 = load int* %tmp1593		; <int> [#uses=1]
	%tmp1594 = cast int %tmp1594 to ubyte		; <ubyte> [#uses=1]
	%tmp1595 = shr ulong %tmp1591, ubyte %tmp1594		; <ulong> [#uses=1]
	%tmp1595 = cast ulong %tmp1595 to uint		; <uint> [#uses=1]
	%tmp1596 = and uint %tmp1595, 255		; <uint> [#uses=1]
	%tmp1598 = getelementptr [64 x [256 x int]]* %bishop_mobility_rr45, int 0, int %tmp1572, uint %tmp1596		; <int*> [#uses=1]
	%tmp1599 = load int* %tmp1598		; <int> [#uses=1]
	%tmp1600.neg = sub int 0, %tmp1588		; <int> [#uses=1]
	%tmp1602 = sub int %tmp1600.neg, %tmp1599		; <int> [#uses=1]
	%tmp1604 = getelementptr [64 x sbyte]* %black_outpost, int 0, int %tmp1572		; <sbyte*> [#uses=1]
	%tmp1605 = load sbyte* %tmp1604		; <sbyte> [#uses=1]
	%tmp1606 = seteq sbyte %tmp1605, 0		; <bool> [#uses=1]
	br bool %tmp1606, label %cond_next1637, label %cond_true1607

cond_true1607:		; preds = %bb1567
	ret void

cond_next1637:		; preds = %bb1567
	%tmp1662 = sub int %tmp1602, 0		; <int> [#uses=0]
	ret void
}
