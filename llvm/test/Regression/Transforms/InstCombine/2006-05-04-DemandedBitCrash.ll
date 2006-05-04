; RUN: llvm-as < %s | opt -instcombine -disable-output

void %test() {
bb38.i:
	%varspec.0.i1014 = cast long 123814269237067777 to ulong		; <ulong> [#uses=1]
	%locspec.0.i1015 = cast int 1 to uint		; <uint> [#uses=2]
	%tmp51391.i1018 = shr ulong %varspec.0.i1014, ubyte 16		; <ulong> [#uses=1]
	%tmp51392.i1019 = cast ulong %tmp51391.i1018 to uint		; <uint> [#uses=2]
	%tmp51392.mask.i1020 = shr uint %tmp51392.i1019, ubyte 29		; <uint> [#uses=1]
	%tmp7.i1021 = and uint %tmp51392.mask.i1020, 1		; <uint> [#uses=2]
	%tmp18.i1026 = shr uint %tmp51392.i1019, ubyte 31		; <uint> [#uses=2]
	%tmp18.i1027 = cast uint %tmp18.i1026 to ubyte		; <ubyte> [#uses=1]
	br bool false, label %cond_false1148.i1653, label %bb377.i1259

bb377.i1259:		; preds = %bb38.i
	br bool false, label %cond_true541.i1317, label %cond_false1148.i1653

cond_true541.i1317:		; preds = %bb377.i1259
	%tmp545.i1318 = shr uint %locspec.0.i1015, ubyte 10		; <uint> [#uses=1]
	%tmp550.i1319 = shr uint %locspec.0.i1015, ubyte 4		; <uint> [#uses=1]
	%tmp550551.i1320 = and uint %tmp550.i1319, 63		; <uint> [#uses=1]
	%tmp553.i1321 = setlt uint %tmp550551.i1320, 4		; <bool> [#uses=1]
	%tmp558.i1322 = seteq uint %tmp7.i1021, 0		; <bool> [#uses=1]
	%bothcond.i1326 = or bool %tmp553.i1321, false		; <bool> [#uses=1]
	%bothcond1.i1327 = or bool %bothcond.i1326, false		; <bool> [#uses=1]
	%bothcond2.not.i1328 = or bool %bothcond1.i1327, false		; <bool> [#uses=1]
	%bothcond3.i1329 = or bool %bothcond2.not.i1328, %tmp558.i1322		; <bool> [#uses=0]
	br bool false, label %cond_true583.i1333, label %cond_next592.i1337

cond_true583.i1333:		; preds = %cond_true541.i1317
	br bool false, label %cond_true586.i1335, label %cond_next592.i1337

cond_true586.i1335:		; preds = %cond_true583.i1333
	br label %cond_true.i

cond_next592.i1337:		; preds = %cond_true583.i1333, %cond_true541.i1317
	%mask_z.0.i1339 = phi uint [ %tmp18.i1026, %cond_true541.i1317 ], [ 0, %cond_true583.i1333 ]		; <uint> [#uses=0]
	%tmp594.i1340 = and uint %tmp545.i1318, 15		; <uint> [#uses=0]
	br label %cond_true.i

cond_false1148.i1653:		; preds = %bb377.i1259, %bb38.i
	%tmp1150.i1654 = seteq uint %tmp7.i1021, 0		; <bool> [#uses=1]
	%tmp1160.i1656 = seteq ubyte %tmp18.i1027, 0		; <bool> [#uses=1]
	%bothcond8.i1658 = or bool %tmp1150.i1654, %tmp1160.i1656		; <bool> [#uses=1]
	%bothcond9.i1659 = or bool %bothcond8.i1658, false		; <bool> [#uses=0]
	br label %cond_true.i

cond_true.i:		; preds = %cond_false1148.i1653, %cond_next592.i1337, %cond_true586.i1335
	ret void
}
