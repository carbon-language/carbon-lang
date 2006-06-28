; RUN: llvm-as < %s | llc
	%struct.rtunion = type { long }
	%struct.rtx_def = type { ushort, ubyte, ubyte, [1 x %struct.rtunion] }
%ix86_cpu = external global uint		; <uint*> [#uses=1]
%which_alternative = external global int		; <int*> [#uses=3]

implementation   ; Functions:

declare fastcc int %recog()

void %athlon_fp_unit_ready_cost() {
entry:
	%tmp = setlt int 0, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true.i, label %cond_true

cond_true:		; preds = %entry
	ret void

cond_true.i:		; preds = %entry
	%tmp8.i = tail call fastcc int %recog( )		; <int> [#uses=1]
	switch int %tmp8.i, label %UnifiedReturnBlock [
		 int -1, label %bb2063
		 int 19, label %bb2035
		 int 20, label %bb2035
		 int 21, label %bb2035
		 int 23, label %bb2035
		 int 24, label %bb2035
		 int 27, label %bb2035
		 int 32, label %bb2035
		 int 33, label %bb1994
		 int 35, label %bb2035
		 int 36, label %bb1994
		 int 90, label %bb1948
		 int 94, label %bb1948
		 int 95, label %bb1948
		 int 101, label %bb1648
		 int 102, label %bb1648
		 int 103, label %bb1648
		 int 104, label %bb1648
		 int 133, label %bb1419
		 int 135, label %bb1238
		 int 136, label %bb1238
		 int 137, label %bb1238
		 int 138, label %bb1238
		 int 139, label %bb1201
		 int 140, label %bb1201
		 int 141, label %bb1154
		 int 142, label %bb1126
		 int 144, label %bb1201
		 int 145, label %bb1126
		 int 146, label %bb1201
		 int 147, label %bb1126
		 int 148, label %bb1201
		 int 149, label %bb1126
		 int 150, label %bb1201
		 int 151, label %bb1126
		 int 152, label %bb1096
		 int 153, label %bb1096
		 int 154, label %bb1096
		 int 157, label %bb1096
		 int 158, label %bb1096
		 int 159, label %bb1096
		 int 162, label %bb1096
		 int 163, label %bb1096
		 int 164, label %bb1096
		 int 167, label %bb1201
		 int 168, label %bb1201
		 int 170, label %bb1201
		 int 171, label %bb1201
		 int 173, label %bb1201
		 int 174, label %bb1201
		 int 176, label %bb1201
		 int 177, label %bb1201
		 int 179, label %bb993
		 int 180, label %bb993
		 int 181, label %bb993
		 int 182, label %bb993
		 int 183, label %bb993
		 int 184, label %bb993
		 int 365, label %bb1126
		 int 366, label %bb1126
		 int 367, label %bb1126
		 int 368, label %bb1126
		 int 369, label %bb1126
		 int 370, label %bb1126
		 int 371, label %bb1126
		 int 372, label %bb1126
		 int 373, label %bb1126
		 int 384, label %bb1126
		 int 385, label %bb1126
		 int 386, label %bb1126
		 int 387, label %bb1126
		 int 388, label %bb1126
		 int 389, label %bb1126
		 int 390, label %bb1126
		 int 391, label %bb1126
		 int 392, label %bb1126
		 int 525, label %bb919
		 int 526, label %bb839
		 int 528, label %bb919
		 int 529, label %bb839
		 int 531, label %cond_next6.i119
		 int 532, label %cond_next6.i97
		 int 533, label %cond_next6.i81
		 int 534, label %bb495
		 int 536, label %cond_next6.i81
		 int 537, label %cond_next6.i81
		 int 538, label %bb396
		 int 539, label %bb288
		 int 541, label %bb396
		 int 542, label %bb396
		 int 543, label %bb396
		 int 544, label %bb396
		 int 545, label %bb189
		 int 546, label %cond_next6.i
		 int 547, label %bb189
		 int 548, label %cond_next6.i
		 int 549, label %bb189
		 int 550, label %cond_next6.i
		 int 551, label %bb189
		 int 552, label %cond_next6.i
		 int 553, label %bb189
		 int 554, label %cond_next6.i
		 int 555, label %bb189
		 int 556, label %cond_next6.i
		 int 557, label %bb189
		 int 558, label %cond_next6.i
		 int 618, label %bb40
		 int 619, label %bb18
		 int 620, label %bb40
		 int 621, label %bb10
		 int 622, label %bb10
	]

bb10:		; preds = %cond_true.i, %cond_true.i
	ret void

bb18:		; preds = %cond_true.i
	ret void

bb40:		; preds = %cond_true.i, %cond_true.i
	ret void

cond_next6.i:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb189:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb288:		; preds = %cond_true.i
	ret void

bb396:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb495:		; preds = %cond_true.i
	ret void

cond_next6.i81:		; preds = %cond_true.i, %cond_true.i, %cond_true.i
	ret void

cond_next6.i97:		; preds = %cond_true.i
	ret void

cond_next6.i119:		; preds = %cond_true.i
	%tmp.i126 = seteq ushort 0, 78		; <bool> [#uses=1]
	br bool %tmp.i126, label %cond_next778, label %bb802

cond_next778:		; preds = %cond_next6.i119
	%tmp781 = seteq uint 0, 1		; <bool> [#uses=1]
	br bool %tmp781, label %cond_next784, label %bb790

cond_next784:		; preds = %cond_next778
	%tmp785 = load uint* %ix86_cpu		; <uint> [#uses=1]
	%tmp786 = seteq uint %tmp785, 5		; <bool> [#uses=1]
	br bool %tmp786, label %UnifiedReturnBlock, label %bb790

bb790:		; preds = %cond_next784, %cond_next778
	%tmp793 = seteq uint 0, 1		; <bool> [#uses=0]
	ret void

bb802:		; preds = %cond_next6.i119
	ret void

bb839:		; preds = %cond_true.i, %cond_true.i
	ret void

bb919:		; preds = %cond_true.i, %cond_true.i
	ret void

bb993:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1096:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1126:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1154:		; preds = %cond_true.i
	ret void

bb1201:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1238:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1419:		; preds = %cond_true.i
	ret void

bb1648:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	%tmp1650 = load int* %which_alternative		; <int> [#uses=1]
	switch int %tmp1650, label %bb1701 [
		 int 0, label %cond_next1675
		 int 1, label %cond_next1675
		 int 2, label %cond_next1675
	]

cond_next1675:		; preds = %bb1648, %bb1648, %bb1648
	ret void

bb1701:		; preds = %bb1648
	%tmp1702 = load int* %which_alternative		; <int> [#uses=1]
	switch int %tmp1702, label %bb1808 [
		 int 0, label %cond_next1727
		 int 1, label %cond_next1727
		 int 2, label %cond_next1727
	]

cond_next1727:		; preds = %bb1701, %bb1701, %bb1701
	ret void

bb1808:		; preds = %bb1701
	%bothcond696 = or bool false, false		; <bool> [#uses=1]
	br bool %bothcond696, label %bb1876, label %cond_next1834

cond_next1834:		; preds = %bb1808
	ret void

bb1876:		; preds = %bb1808
	%tmp1877 = load int* %which_alternative		; <int> [#uses=4]
	%tmp1877 = cast int %tmp1877 to uint		; <uint> [#uses=1]
	%bothcond699 = setlt uint %tmp1877, 2		; <bool> [#uses=1]
	%tmp1888 = seteq int %tmp1877, 2		; <bool> [#uses=1]
	%bothcond700 = or bool %bothcond699, %tmp1888		; <bool> [#uses=1]
	%bothcond700.not = xor bool %bothcond700, true		; <bool> [#uses=1]
	%tmp1894 = seteq int %tmp1877, 3		; <bool> [#uses=1]
	%bothcond701 = or bool %tmp1894, %bothcond700.not		; <bool> [#uses=1]
	%bothcond702 = or bool %bothcond701, false		; <bool> [#uses=1]
	br bool %bothcond702, label %UnifiedReturnBlock, label %cond_next1902

cond_next1902:		; preds = %bb1876
	switch int %tmp1877, label %cond_next1937 [
		 int 0, label %bb1918
		 int 1, label %bb1918
		 int 2, label %bb1918
	]

bb1918:		; preds = %cond_next1902, %cond_next1902, %cond_next1902
	ret void

cond_next1937:		; preds = %cond_next1902
	ret void

bb1948:		; preds = %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb1994:		; preds = %cond_true.i, %cond_true.i
	ret void

bb2035:		; preds = %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i, %cond_true.i
	ret void

bb2063:		; preds = %cond_true.i
	ret void

UnifiedReturnBlock:		; preds = %bb1876, %cond_next784, %cond_true.i
	%UnifiedRetVal = phi int [ 100, %bb1876 ], [ 100, %cond_true.i ], [ 4, %cond_next784 ]		; <int> [#uses=0]
	ret void
}
