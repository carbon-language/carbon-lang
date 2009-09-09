; RUN: llc < %s -march=bfin

; The switch expansion uses a dynamic shl, and it produces a jumptable

define void @athlon_fp_unit_ready_cost() {
entry:
	switch i32 0, label %UnifiedReturnBlock [
		i32 -1, label %bb2063
		i32 19, label %bb2035
		i32 20, label %bb2035
		i32 21, label %bb2035
		i32 23, label %bb2035
		i32 24, label %bb2035
		i32 27, label %bb2035
		i32 32, label %bb2035
		i32 33, label %bb1994
		i32 35, label %bb2035
		i32 36, label %bb1994
		i32 90, label %bb1948
		i32 94, label %bb1948
		i32 95, label %bb1948
		i32 133, label %bb1419
		i32 135, label %bb1238
		i32 136, label %bb1238
		i32 137, label %bb1238
		i32 138, label %bb1238
		i32 139, label %bb1201
		i32 140, label %bb1201
		i32 141, label %bb1154
		i32 142, label %bb1126
		i32 144, label %bb1201
		i32 145, label %bb1126
		i32 146, label %bb1201
		i32 147, label %bb1126
		i32 148, label %bb1201
		i32 149, label %bb1126
		i32 150, label %bb1201
		i32 151, label %bb1126
		i32 152, label %bb1096
		i32 153, label %bb1096
		i32 154, label %bb1096
		i32 157, label %bb1096
		i32 158, label %bb1096
		i32 159, label %bb1096
		i32 162, label %bb1096
		i32 163, label %bb1096
		i32 164, label %bb1096
		i32 167, label %bb1201
		i32 168, label %bb1201
		i32 170, label %bb1201
		i32 171, label %bb1201
		i32 173, label %bb1201
		i32 174, label %bb1201
		i32 176, label %bb1201
		i32 177, label %bb1201
		i32 179, label %bb993
		i32 180, label %bb993
		i32 181, label %bb993
		i32 182, label %bb993
		i32 183, label %bb993
		i32 184, label %bb993
		i32 365, label %bb1126
		i32 366, label %bb1126
		i32 367, label %bb1126
		i32 368, label %bb1126
		i32 369, label %bb1126
		i32 370, label %bb1126
		i32 371, label %bb1126
		i32 372, label %bb1126
		i32 373, label %bb1126
		i32 384, label %bb1126
		i32 385, label %bb1126
		i32 386, label %bb1126
		i32 387, label %bb1126
		i32 388, label %bb1126
		i32 389, label %bb1126
		i32 390, label %bb1126
		i32 391, label %bb1126
		i32 392, label %bb1126
		i32 525, label %bb919
		i32 526, label %bb839
		i32 528, label %bb919
		i32 529, label %bb839
		i32 532, label %cond_next6.i97
		i32 533, label %cond_next6.i81
		i32 534, label %bb495
		i32 536, label %cond_next6.i81
		i32 537, label %cond_next6.i81
		i32 538, label %bb396
		i32 539, label %bb288
		i32 541, label %bb396
		i32 542, label %bb396
		i32 543, label %bb396
		i32 544, label %bb396
		i32 545, label %bb189
		i32 546, label %cond_next6.i
		i32 547, label %bb189
		i32 548, label %cond_next6.i
		i32 549, label %bb189
		i32 550, label %cond_next6.i
		i32 551, label %bb189
		i32 552, label %cond_next6.i
		i32 553, label %bb189
		i32 554, label %cond_next6.i
		i32 555, label %bb189
		i32 556, label %cond_next6.i
		i32 557, label %bb189
		i32 558, label %cond_next6.i
		i32 618, label %bb40
		i32 619, label %bb18
		i32 620, label %bb40
		i32 621, label %bb10
		i32 622, label %bb10
	]

bb10:
	ret void

bb18:
	ret void

bb40:
	ret void

cond_next6.i:
	ret void

bb189:
	ret void

bb288:
	ret void

bb396:
	ret void

bb495:
	ret void

cond_next6.i81:
	ret void

cond_next6.i97:
	ret void

bb839:
	ret void

bb919:
	ret void

bb993:
	ret void

bb1096:
	ret void

bb1126:
	ret void

bb1154:
	ret void

bb1201:
	ret void

bb1238:
	ret void

bb1419:
	ret void

bb1948:
	ret void

bb1994:
	ret void

bb2035:
	ret void

bb2063:
	ret void

UnifiedReturnBlock:
	ret void
}
