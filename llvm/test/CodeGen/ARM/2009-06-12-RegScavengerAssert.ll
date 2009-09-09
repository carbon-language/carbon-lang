; RUN: llc < %s -mtriple=armv6-apple-darwin

	type { i32, i32, %struct.D_Sym**, [3 x %struct.D_Sym*] }		; type %0
	type { i32, %struct.D_Reduction** }		; type %1
	type { i32, %struct.D_RightEpsilonHint* }		; type %2
	type { i32, %struct.D_ErrorRecoveryHint* }		; type %3
	type { i32, i32, %struct.D_Reduction**, [3 x %struct.D_Reduction*] }		; type %4
	%struct.D_ErrorRecoveryHint = type { i16, i16, i8* }
	%struct.D_ParseNode = type { i32, %struct.d_loc_t, i8*, i8*, %struct.D_Scope*, void (%struct.D_Parser*, %struct.d_loc_t*, i8**)*, i8*, i8* }
	%struct.D_Parser = type { i8*, void (%struct.D_Parser*, %struct.d_loc_t*, i8**)*, %struct.D_Scope*, void (%struct.D_Parser*)*, %struct.D_ParseNode* (%struct.D_Parser*, i32, %struct.D_ParseNode**)*, void (%struct.D_ParseNode*)*, %struct.d_loc_t, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.D_ParserTables = type { i32, %struct.D_State*, i16*, i32, i32, %struct.D_Symbol*, void (%struct.D_Parser*, %struct.d_loc_t*, i8**)*, i32, %struct.D_Pass*, i32 }
	%struct.D_Pass = type { i8*, i32, i32, i32 }
	%struct.D_Reduction = type { i16, i16, i32 (i8*, i8**, i32, i32, %struct.D_Parser*)*, i32 (i8*, i8**, i32, i32, %struct.D_Parser*)*, i16, i16, i32, i32, i32, i32, i32 (i8*, i8**, i32, i32, %struct.D_Parser*)** }
	%struct.D_RightEpsilonHint = type { i16, i16, %struct.D_Reduction* }
	%struct.D_Scope = type { i8, %struct.D_Sym*, %struct.D_SymHash*, %struct.D_Sym*, %struct.D_Scope*, %struct.D_Scope*, %struct.D_Scope*, %struct.D_Scope*, %struct.D_Scope* }
	%struct.D_Shift = type { i16, i8, i8, i32, i32, i32 (i8*, i8**, i32, i32, %struct.D_Parser*)* }
	%struct.D_State = type { i8*, i32, %1, %2, %3, %struct.D_Shift**, i32 (i8**, i32*, i32*, i16*, i32*, i8*, i32*)*, i8*, i8, i8, i8, i8*, %struct.D_Shift***, i32 }
	%struct.D_Sym = type { i8*, i32, i32, %struct.D_Sym*, %struct.D_Sym*, i32 }
	%struct.D_SymHash = type { i32, i32, %0 }
	%struct.D_Symbol = type { i32, i8*, i32 }
	%struct.PNode = type { i32, i32, i32, i32, %struct.D_Reduction*, %struct.D_Shift*, i32, %struct.VecPNode, i32, i8, i8, %struct.PNode*, %struct.PNode*, %struct.PNode*, %struct.PNode*, i8*, i8*, %struct.D_Scope*, i8*, %struct.D_ParseNode }
	%struct.PNodeHash = type { %struct.PNode**, i32, i32, i32, %struct.PNode* }
	%struct.Parser = type { %struct.D_Parser, i8*, i8*, %struct.D_ParserTables*, i32, i32, i32, i32, i32, i32, i32, %struct.PNodeHash, %struct.SNodeHash, %struct.Reduction*, %struct.Shift*, %struct.D_Scope*, %struct.SNode*, i32, %struct.Reduction*, %struct.Shift*, i32, %struct.PNode*, %struct.SNode*, %struct.ZNode*, %4, %struct.ShiftResult*, %struct.D_Shift, %struct.Parser*, i8* }
	%struct.Reduction = type { %struct.ZNode*, %struct.SNode*, %struct.D_Reduction*, %struct.SNode*, i32, %struct.Reduction* }
	%struct.SNode = type { %struct.D_State*, %struct.D_Scope*, i8*, %struct.d_loc_t, i32, %struct.PNode*, %struct.VecZNode, i32, %struct.SNode*, %struct.SNode* }
	%struct.SNodeHash = type { %struct.SNode**, i32, i32, i32, %struct.SNode*, %struct.SNode* }
	%struct.Shift = type { %struct.SNode*, %struct.Shift* }
	%struct.ShiftResult = type { %struct.D_Shift*, %struct.d_loc_t }
	%struct.VecPNode = type { i32, i32, %struct.PNode**, [3 x %struct.PNode*] }
	%struct.VecSNode = type { i32, i32, %struct.SNode**, [3 x %struct.SNode*] }
	%struct.VecZNode = type { i32, i32, %struct.ZNode**, [3 x %struct.ZNode*] }
	%struct.ZNode = type { %struct.PNode*, %struct.VecSNode }
	%struct.d_loc_t = type { i8*, i8*, i32, i32, i32 }

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

define fastcc i32 @exhaustive_parse(%struct.Parser* %p, i32 %state) nounwind {
entry:
	store i8* undef, i8** undef, align 4
	%0 = getelementptr %struct.Parser* %p, i32 0, i32 0, i32 6		; <%struct.d_loc_t*> [#uses=1]
	%1 = bitcast %struct.d_loc_t* %0 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32(i8* undef, i8* %1, i32 20, i32 4)
	br label %bb10

bb10:		; preds = %bb30, %bb29, %bb26, %entry
	br i1 undef, label %bb18, label %bb20

bb18:		; preds = %bb10
	br i1 undef, label %bb20, label %bb19

bb19:		; preds = %bb18
	br label %bb20

bb20:		; preds = %bb19, %bb18, %bb10
	br i1 undef, label %bb21, label %bb22

bb21:		; preds = %bb20
	unreachable

bb22:		; preds = %bb20
	br i1 undef, label %bb24, label %bb26

bb24:		; preds = %bb22
	unreachable

bb26:		; preds = %bb22
	br i1 undef, label %bb10, label %bb29

bb29:		; preds = %bb26
	br i1 undef, label %bb10, label %bb30

bb30:		; preds = %bb29
	br i1 undef, label %bb31, label %bb10

bb31:		; preds = %bb30
	unreachable
}
