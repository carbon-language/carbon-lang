; RUN: llvm-upgrade < %s | llvm-as | llc 

	%struct.CUMULATIVE_ARGS = type { int, int, int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.VEC_edge = type { uint, uint, [1 x %struct.edge_def*] }
	%struct._obstack_chunk = type { sbyte*, %struct._obstack_chunk*, [4 x sbyte] }
	%struct.basic_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.tree_node*, %struct.VEC_edge*, %struct.VEC_edge*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, sbyte*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.reorder_block_def*, %struct.bb_ann_d*, long, int, int, int, int }
	%struct.bb_ann_d = type { %struct.tree_node*, ubyte, %struct.edge_prediction* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, uint, [4 x uint] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, uint, %struct.bitmap_obstack* }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.cost_pair = type { %struct.iv_cand*, uint, %struct.bitmap_head_def* }
	%struct.dataflow_d = type { %struct.varray_head_tag*, [2 x %struct.tree_node*] }
	%struct.def_operand_ptr = type { %struct.tree_node** }
	%struct.def_optype_d = type { uint, [1 x %struct.def_operand_ptr] }
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, sbyte*, %struct.location_t*, int, int, long, uint }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_prediction = type { %struct.edge_prediction*, %struct.edge_def*, uint, int }
	%struct.eh_status = type opaque
	%struct.emit_status = type { int, int, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, int, %struct.location_t, int, ubyte*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { int, int, int, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, int, int, int, int, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, ubyte, int, long, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, int, %struct.var_refs_queue*, int, int, %struct.rtvec_def*, %struct.tree_node*, int, int, int, %struct.machine_function*, uint, uint, bool, bool, %struct.language_function*, %struct.rtx_def*, uint, int, int, int, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, ubyte, ubyte, ubyte }
	%struct.htab = type { uint (sbyte*)*, int (sbyte*, sbyte*)*, void (sbyte*)*, sbyte**, uint, uint, uint, uint, uint, sbyte* (uint, uint)*, void (sbyte*)*, sbyte*, sbyte* (sbyte*, uint, uint)*, void (sbyte*, sbyte*)*, uint }
	%struct.initial_value_struct = type opaque
	%struct.iv = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, bool, bool, uint }
	%struct.iv_cand = type { uint, bool, uint, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.iv*, uint }
	%struct.iv_use = type { uint, uint, %struct.iv*, %struct.tree_node*, %struct.tree_node**, %struct.bitmap_head_def*, uint, %struct.cost_pair*, %struct.iv_cand* }
	%struct.ivopts_data = type { %struct.loop*, %struct.htab*, uint, %struct.version_info*, %struct.bitmap_head_def*, uint, %struct.varray_head_tag*, %struct.varray_head_tag*, %struct.bitmap_head_def*, bool }
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { sbyte*, int }
	%struct.loop = type { int, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.lpt_decision, uint, uint, %struct.edge_def**, int, %struct.basic_block_def*, %struct.basic_block_def*, uint, %struct.edge_def**, int, %struct.edge_def**, int, %struct.simple_bitmap_def*, int, %struct.loop**, int, %struct.loop*, %struct.loop*, %struct.loop*, %struct.loop*, int, sbyte*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, int, %struct.tree_node*, %struct.tree_node*, %struct.nb_iter_bound*, %struct.edge_def*, bool }
	%struct.lpt_decision = type { uint, uint }
	%struct.machine_function = type { %struct.stack_local_entry*, sbyte*, %struct.rtx_def*, int, int, int, int, int }
	%struct.nb_iter_bound = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.nb_iter_bound* }
	%struct.obstack = type { int, %struct._obstack_chunk*, sbyte*, sbyte*, sbyte*, int, int, %struct._obstack_chunk* (sbyte*, int)*, void (sbyte*, %struct._obstack_chunk*)*, sbyte*, ubyte }
	%struct.reorder_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, int, int, int }
	%struct.rtvec_def = type { int, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { ushort, ubyte, ubyte, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.simple_bitmap_def = type { uint, uint, uint, [1 x ulong] }
	%struct.stack_local_entry = type opaque
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, ubyte, %struct.basic_block_def*, %struct.stmt_operands_d, %struct.dataflow_d*, %struct.bitmap_head_def*, uint }
	%struct.stmt_operands_d = type { %struct.def_optype_d*, %struct.def_optype_d*, %struct.v_may_def_optype_d*, %struct.vuse_optype_d*, %struct.v_may_def_optype_d* }
	%struct.temp_slot = type opaque
	%struct.tree_ann_common_d = type { uint, sbyte*, %struct.tree_node* }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_ann_d*, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, uint, %struct.tree_node*, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, uint, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, int, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, long, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { long }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x long] }
	%struct.v_def_use_operand_type_t = type { %struct.tree_node*, %struct.tree_node* }
	%struct.v_may_def_optype_d = type { uint, [1 x %struct.v_def_use_operand_type_t] }
	%struct.var_refs_queue = type { %struct.rtx_def*, uint, int, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { uint, uint, uint, sbyte*, %struct.u }
	%struct.version_info = type { %struct.tree_node*, %struct.iv*, bool, uint, bool }
	%struct.vuse_optype_d = type { uint, [1 x %struct.tree_node*] }

implementation   ; Functions:

bool %determine_use_iv_cost(%struct.ivopts_data* %data, %struct.iv_use* %use, %struct.iv_cand* %cand) {
entry:
	switch uint 0, label %bb91 [
		 uint 0, label %bb
		 uint 1, label %bb6
		 uint 3, label %cond_next135
	]

bb:		; preds = %entry
	ret bool false

bb6:		; preds = %entry
	br bool false, label %bb87, label %cond_next27

cond_next27:		; preds = %bb6
	br bool false, label %cond_true30, label %cond_next55

cond_true30:		; preds = %cond_next27
	br bool false, label %cond_next41, label %cond_true35

cond_true35:		; preds = %cond_true30
	ret bool false

cond_next41:		; preds = %cond_true30
	%tmp44 = call uint %force_var_cost( %struct.ivopts_data* %data, %struct.tree_node* null, %struct.bitmap_head_def** null )		; <uint> [#uses=2]
	%tmp46 = div uint %tmp44, 5		; <uint> [#uses=1]
	call void %set_use_iv_cost( %struct.ivopts_data* %data, %struct.iv_use* %use, %struct.iv_cand* %cand, uint %tmp46, %struct.bitmap_head_def* null )
	%tmp44.off = add uint %tmp44, 4244967296		; <uint> [#uses=1]
	%tmp52 = setgt uint %tmp44.off, 4		; <bool> [#uses=1]
	%tmp52 = cast bool %tmp52 to int		; <int> [#uses=1]
	br label %bb87

cond_next55:		; preds = %cond_next27
	ret bool false

bb87:		; preds = %cond_next41, %bb6
	%tmp2.0 = phi int [ %tmp52, %cond_next41 ], [ 1, %bb6 ]		; <int> [#uses=0]
	ret bool false

bb91:		; preds = %entry
	ret bool false

cond_next135:		; preds = %entry
	%tmp193 = call bool %determine_use_iv_cost_generic( %struct.ivopts_data* %data, %struct.iv_use* %use, %struct.iv_cand* %cand )		; <bool> [#uses=0]
	ret bool false
}

declare void %set_use_iv_cost(%struct.ivopts_data*, %struct.iv_use*, %struct.iv_cand*, uint, %struct.bitmap_head_def*)

declare uint %force_var_cost(%struct.ivopts_data*, %struct.tree_node*, %struct.bitmap_head_def**)

declare bool %determine_use_iv_cost_generic(%struct.ivopts_data*, %struct.iv_use*, %struct.iv_cand*)
