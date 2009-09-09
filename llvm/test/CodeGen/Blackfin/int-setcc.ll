; RUN: llc < %s -march=bfin -verify-machineinstrs > %t

define fastcc void @Evaluate() {
entry:
	br i1 false, label %cond_false186, label %cond_true

cond_true:		; preds = %entry
	ret void

cond_false186:		; preds = %entry
	br i1 false, label %cond_true293, label %bb203

bb203:		; preds = %cond_false186
	ret void

cond_true293:		; preds = %cond_false186
	br i1 false, label %cond_true298, label %cond_next317

cond_true298:		; preds = %cond_true293
	br i1 false, label %cond_next518, label %cond_true397.preheader

cond_next317:		; preds = %cond_true293
	ret void

cond_true397.preheader:		; preds = %cond_true298
	ret void

cond_next518:		; preds = %cond_true298
	br i1 false, label %bb1069, label %cond_true522

cond_true522:		; preds = %cond_next518
	ret void

bb1069:		; preds = %cond_next518
	br i1 false, label %cond_next1131, label %bb1096

bb1096:		; preds = %bb1069
	ret void

cond_next1131:		; preds = %bb1069
	br i1 false, label %cond_next1207, label %cond_true1150

cond_true1150:		; preds = %cond_next1131
	ret void

cond_next1207:		; preds = %cond_next1131
	br i1 false, label %cond_next1219, label %cond_true1211

cond_true1211:		; preds = %cond_next1207
	ret void

cond_next1219:		; preds = %cond_next1207
	br i1 false, label %cond_true1223, label %cond_next1283

cond_true1223:		; preds = %cond_next1219
	br i1 false, label %cond_true1254, label %cond_true1264

cond_true1254:		; preds = %cond_true1223
	br i1 false, label %bb1567, label %cond_true1369.preheader

cond_true1264:		; preds = %cond_true1223
	ret void

cond_next1283:		; preds = %cond_next1219
	ret void

cond_true1369.preheader:		; preds = %cond_true1254
	ret void

bb1567:		; preds = %cond_true1254
	%tmp1605 = load i8* null		; <i8> [#uses=1]
	%tmp1606 = icmp eq i8 %tmp1605, 0		; <i1> [#uses=1]
	br i1 %tmp1606, label %cond_next1637, label %cond_true1607

cond_true1607:		; preds = %bb1567
	ret void

cond_next1637:		; preds = %bb1567
	ret void
}
