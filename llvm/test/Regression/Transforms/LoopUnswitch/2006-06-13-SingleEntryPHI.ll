; RUN: llvm-as < %s | opt -loop-unswitch -disable-output

	%struct.BLEND_MAP = type { short, short, short, int, %struct.BLEND_MAP_ENTRY* }
	%struct.BLEND_MAP_ENTRY = type { float, ubyte, { [5 x float], [4 x ubyte] } }
	%struct.TPATTERN = type { ushort, ushort, ushort, int, float, float, float, %struct.WARP*, %struct.TPATTERN*, %struct.BLEND_MAP*, { %struct.anon, [4 x ubyte] } }
	%struct.TURB = type { ushort, %struct.WARP*, [3 x double], int, float, float }
	%struct.WARP = type { ushort, %struct.WARP* }
	%struct.anon = type { float, [3 x double] }

implementation   ; Functions:

void %Parse_Pattern() {
entry:
	br label %bb1096.outer20

bb671:		; preds = %cond_true1099
	br label %bb1096.outer23

bb1096.outer20.loopexit:		; preds = %cond_true1099
	%Local_Turb.0.ph24.lcssa = phi %struct.TURB* [ %Local_Turb.0.ph24, %cond_true1099 ]		; <%struct.TURB*> [#uses=1]
	br label %bb1096.outer20

bb1096.outer20:		; preds = %bb1096.outer20.loopexit, %entry
	%Local_Turb.0.ph22 = phi %struct.TURB* [ undef, %entry ], [ %Local_Turb.0.ph24.lcssa, %bb1096.outer20.loopexit ]		; <%struct.TURB*> [#uses=1]
	%tmp1098 = seteq int 0, 0		; <bool> [#uses=1]
	br label %bb1096.outer23

bb1096.outer23:		; preds = %bb1096.outer20, %bb671
	%Local_Turb.0.ph24 = phi %struct.TURB* [ %Local_Turb.0.ph22, %bb1096.outer20 ], [ null, %bb671 ]		; <%struct.TURB*> [#uses=2]
	br label %bb1096

bb1096:		; preds = %cond_true1099, %bb1096.outer23
	br bool %tmp1098, label %cond_true1099, label %bb1102

cond_true1099:		; preds = %bb1096
	switch int 0, label %bb1096.outer20.loopexit [
		 int 161, label %bb671
		 int 359, label %bb1096
	]

bb1102:		; preds = %bb1096
	%Local_Turb.0.ph24.lcssa1 = phi %struct.TURB* [ %Local_Turb.0.ph24, %bb1096 ]		; <%struct.TURB*> [#uses=0]
	ret void
}
