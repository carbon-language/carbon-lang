; RUN: opt < %s -inline -reassociate -loop-rotate -loop-index-split -indvars -simplifycfg -verify
; PR4471

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.VEC_basic_block_base = type { i32, i32, [1 x %struct.basic_block_def*] }
	%struct.VEC_basic_block_gc = type { %struct.VEC_basic_block_base }
	%struct.VEC_edge_base = type { i32, i32, [1 x %struct.edge_def*] }
	%struct.VEC_edge_gc = type { %struct.VEC_edge_base }
	%struct.VEC_rtx_base = type { i32, i32, [1 x %struct.rtx_def*] }
	%struct.VEC_rtx_gc = type { %struct.VEC_rtx_base }
	%struct.VEC_temp_slot_p_base = type { i32, i32, [1 x %struct.temp_slot*] }
	%struct.VEC_temp_slot_p_gc = type { %struct.VEC_temp_slot_p_base }
	%struct.VEC_tree_base = type { i32, i32, [1 x %struct.tree_node*] }
	%struct.VEC_tree_gc = type { %struct.VEC_tree_base }
	%struct.__sbuf = type { i8*, i32 }
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct.basic_block_def = type { %struct.tree_node*, %struct.VEC_edge_gc*, %struct.VEC_edge_gc*, i8*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_il_dependent, %struct.tree_node*, %struct.edge_prediction*, i64, i32, i32, i32, i32 }
	%struct.basic_block_il_dependent = type { %struct.rtl_bb_info* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, [2 x i64] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, %struct.bitmap_obstack* }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.block_symbol = type { [3 x %struct.rtunion], %struct.object_block*, i64 }
	%struct.case_node = type { %struct.case_node*, %struct.case_node*, %struct.case_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.control_flow_graph = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.VEC_basic_block_gc*, i32, i32, i32, %struct.VEC_basic_block_gc*, i32 }
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, i8*, %struct.__sbuf*, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_prediction = type opaque
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.__sbuf, i32, i8*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.control_flow_graph*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.VEC_temp_slot_p_gc*, %struct.temp_slot*, %struct.var_refs_queue*, i32, i32, i32, i32, %struct.machine_function*, i32, i32, %struct.language_function*, %struct.htab*, %struct.rtx_def*, i32, i32, %struct.__sbuf, %struct.VEC_tree_gc*, %struct.tree_node*, i8*, i8*, i8*, i8*, i8*, %struct.tree_node*, i8, i8, i8, i8, i8 }
	%struct.htab = type { i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*, i8**, i64, i64, i64, i32, i32, i8* (i64, i64)*, void (i8*)*, i8*, i8* (i8*, i64, i64)*, void (i8*, i8*)*, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.loop = type opaque
	%struct.machine_function = type { %struct.stack_local_entry*, i8*, %struct.rtx_def*, i32, i32, [4 x i32], i32, i32, i32 }
	%struct.object_block = type { %struct.section*, i32, i64, %struct.VEC_rtx_gc*, %struct.VEC_rtx_gc* }
	%struct.obstack = type { i64, %struct._obstack_chunk*, i8*, i8*, i8*, i64, i32, %struct._obstack_chunk* (i8*, i64)*, void (i8*, %struct._obstack_chunk*)*, i8*, i8 }
	%struct.omp_clause_subcode = type { i32 }
	%struct.rtl_bb_info = type { %struct.rtx_def*, %struct.rtx_def*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, %struct.rtx_def*, %struct.rtx_def*, i32 }
	%struct.rtunion = type { i8* }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.section = type { %struct.unnamed_section }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.stack_local_entry = type opaque
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8 }
	%struct.tree_decl_common = type { %struct.tree_decl_minimal, %struct.tree_node*, i8, i8, i8, i8, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_minimal = type { %struct.tree_common, %struct.__sbuf, i32, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl_non_common = type { %struct.tree_decl_with_vis, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_with_rtl = type { %struct.tree_decl_common, %struct.rtx_def* }
	%struct.tree_decl_with_vis = type { %struct.tree_decl_with_rtl, %struct.tree_node*, %struct.tree_node*, i8, i8, i8 }
	%struct.tree_function_decl = type { %struct.tree_decl_non_common, i8, i8, %struct.function* }
	%struct.tree_node = type { %struct.tree_function_decl }
	%struct.u = type { %struct.block_symbol }
	%struct.unnamed_section = type { %struct.omp_clause_subcode, void (i8*)*, i8*, %struct.section* }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%union.tree_ann_d = type opaque

define void @emit_case_bit_tests(%struct.tree_node* %index_type, %struct.tree_node* %index_expr, %struct.tree_node* %minval, %struct.tree_node* %range, %struct.case_node* %nodes, %struct.rtx_def* %default_label) nounwind {
entry:
	br label %bb17

bb:		; preds = %bb17
	%0 = call i64 @tree_low_cst(%struct.tree_node* undef, i32 1) nounwind		; <i64> [#uses=1]
	%1 = trunc i64 %0 to i32		; <i32> [#uses=1]
	br label %bb15

bb10:		; preds = %bb15
	%2 = icmp ugt i32 %j.0, 63		; <i1> [#uses=1]
	br i1 %2, label %bb11, label %bb12

bb11:		; preds = %bb10
	%3 = zext i32 0 to i64		; <i64> [#uses=0]
	br label %bb14

bb12:		; preds = %bb10
	%4 = or i64 undef, undef		; <i64> [#uses=0]
	br label %bb14

bb14:		; preds = %bb12, %bb11
	%5 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %bb15

bb15:		; preds = %bb14, %bb
	%j.0 = phi i32 [ %1, %bb ], [ %5, %bb14 ]		; <i32> [#uses=3]
	%6 = icmp ugt i32 %j.0, undef		; <i1> [#uses=1]
	br i1 %6, label %bb16, label %bb10

bb16:		; preds = %bb15
	br label %bb17

bb17:		; preds = %bb16, %entry
	br i1 undef, label %bb18, label %bb

bb18:		; preds = %bb17
	unreachable
}

declare i64 @tree_low_cst(%struct.tree_node*, i32)

define void @expand_case(%struct.tree_node* %exp) nounwind {
entry:
	br i1 undef, label %bb2, label %bb

bb:		; preds = %entry
	unreachable

bb2:		; preds = %entry
	br i1 undef, label %bb3, label %bb4

bb3:		; preds = %bb2
	unreachable

bb4:		; preds = %bb2
	br i1 undef, label %bb127, label %bb5

bb5:		; preds = %bb4
	br i1 undef, label %bb6, label %bb7

bb6:		; preds = %bb5
	unreachable

bb7:		; preds = %bb5
	br i1 undef, label %bb9, label %bb8

bb8:		; preds = %bb7
	unreachable

bb9:		; preds = %bb7
	br i1 undef, label %bb11, label %bb10

bb10:		; preds = %bb9
	unreachable

bb11:		; preds = %bb9
	br i1 undef, label %bb37, label %bb21

bb21:		; preds = %bb11
	unreachable

bb37:		; preds = %bb11
	br i1 undef, label %bb38, label %bb39

bb38:		; preds = %bb37
	ret void

bb39:		; preds = %bb37
	br i1 undef, label %bb59, label %bb40

bb40:		; preds = %bb39
	br i1 undef, label %bb41, label %bb59

bb41:		; preds = %bb40
	br i1 undef, label %bb42, label %bb59

bb42:		; preds = %bb41
	br i1 undef, label %bb43, label %bb59

bb43:		; preds = %bb42
	br i1 undef, label %bb59, label %bb44

bb44:		; preds = %bb43
	br i1 undef, label %bb56, label %bb58

bb56:		; preds = %bb44
	unreachable

bb58:		; preds = %bb44
	call void @emit_case_bit_tests(%struct.tree_node* undef, %struct.tree_node* undef, %struct.tree_node* null, %struct.tree_node* undef, %struct.case_node* undef, %struct.rtx_def* undef) nounwind
	br i1 undef, label %bb126, label %bb125

bb59:		; preds = %bb43, %bb42, %bb41, %bb40, %bb39
	br i1 undef, label %bb70, label %bb60

bb60:		; preds = %bb59
	unreachable

bb70:		; preds = %bb59
	unreachable

bb125:		; preds = %bb58
	unreachable

bb126:		; preds = %bb58
	unreachable

bb127:		; preds = %bb4
	ret void
}
