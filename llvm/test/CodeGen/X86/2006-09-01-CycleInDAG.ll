; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86

target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8"
	%struct.CUMULATIVE_ARGS = type { int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.FILE = type { ubyte*, int, int, short, short, %struct.__sbuf, int, sbyte*, int (sbyte*)*, int (sbyte*, sbyte*, int)*, long (sbyte*, long, int)*, int (sbyte*, sbyte*, int)*, %struct.__sbuf, %struct.__sFILEX*, int, [3 x ubyte], [1 x ubyte], %struct.__sbuf, int, long }
	%struct.VEC_edge = type { uint, uint, [1 x %struct.edge_def*] }
	%struct.VEC_tree = type { uint, uint, [1 x %struct.tree_node*] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ubyte*, int }
	%struct._obstack_chunk = type { sbyte*, %struct._obstack_chunk*, [4 x sbyte] }
	%struct._var_map = type { %struct.partition_def*, int*, int*, %struct.tree_node**, uint, uint, int* }
	%struct.basic_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.tree_node*, %struct.VEC_edge*, %struct.VEC_edge*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, sbyte*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.reorder_block_def*, %struct.bb_ann_d*, long, int, int, int, int }
	%struct.bb_ann_d = type { %struct.tree_node*, ubyte, %struct.edge_prediction* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, uint, [4 x uint] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, uint, %struct.bitmap_obstack* }
	%struct.bitmap_iterator = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, uint, uint }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.block_stmt_iterator = type { %struct.tree_stmt_iterator, %struct.basic_block_def* }
	%struct.coalesce_list_d = type { %struct._var_map*, %struct.partition_pair_d**, bool }
	%struct.conflict_graph_def = type opaque
	%struct.dataflow_d = type { %struct.varray_head_tag*, [2 x %struct.tree_node*] }
	%struct.def_operand_ptr = type { %struct.tree_node** }
	%struct.def_optype_d = type { uint, [1 x %struct.def_operand_ptr] }
	%struct.die_struct = type opaque
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, sbyte*, %struct.location_t*, int, int, long, uint }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_iterator = type { uint, %struct.VEC_edge** }
	%struct.edge_prediction = type { %struct.edge_prediction*, %struct.edge_def*, uint, int }
	%struct.eh_status = type opaque
	%struct.elt_list = type opaque
	%struct.emit_status = type { int, int, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, int, %struct.location_t, int, ubyte*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { int, int, int, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, int, int, int, int, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, ubyte, int, long, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, int, %struct.var_refs_queue*, int, int, %struct.rtvec_def*, %struct.tree_node*, int, int, int, %struct.machine_function*, uint, uint, bool, bool, %struct.language_function*, %struct.rtx_def*, uint, int, int, int, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, ubyte, ubyte, ubyte }
	%struct.ht_identifier = type { ubyte*, uint, uint }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { sbyte*, int }
	%struct.loop = type opaque
	%struct.machine_function = type { int, uint, sbyte*, int, int }
	%struct.obstack = type { int, %struct._obstack_chunk*, sbyte*, sbyte*, sbyte*, int, int, %struct._obstack_chunk* (sbyte*, int)*, void (sbyte*, %struct._obstack_chunk*)*, sbyte*, ubyte }
	%struct.partition_def = type { int, [1 x %struct.partition_elem] }
	%struct.partition_elem = type { int, %struct.partition_elem*, uint }
	%struct.partition_pair_d = type { int, int, int, %struct.partition_pair_d* }
	%struct.phi_arg_d = type { %struct.tree_node*, bool }
	%struct.pointer_set_t = type opaque
	%struct.ptr_info_def = type { ubyte, %struct.bitmap_head_def*, %struct.tree_node* }
	%struct.real_value = type opaque
	%struct.reg_info_def = type opaque
	%struct.reorder_block_def = type { %struct.rtx_def*, %struct.rtx_def*, %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_def*, int, int, int }
	%struct.rtvec_def = type opaque
	%struct.rtx_def = type opaque
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.simple_bitmap_def = type { uint, uint, uint, [1 x ulong] }
	%struct.ssa_op_iter = type { int, int, int, int, int, int, int, int, int, int, int, int, int, int, %struct.stmt_operands_d*, bool }
	%struct.stmt_ann_d = type { %struct.tree_ann_common_d, ubyte, %struct.basic_block_def*, %struct.stmt_operands_d, %struct.dataflow_d*, %struct.bitmap_head_def*, uint }
	%struct.stmt_operands_d = type { %struct.def_optype_d*, %struct.def_optype_d*, %struct.v_may_def_optype_d*, %struct.vuse_optype_d*, %struct.v_may_def_optype_d* }
	%struct.temp_slot = type opaque
	%struct.tree_ann_common_d = type { uint, sbyte*, %struct.tree_node* }
	%struct.tree_ann_d = type { %struct.stmt_ann_d }
	%struct.tree_binfo = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.VEC_tree }
	%struct.tree_block = type { %struct.tree_common, ubyte, [3 x ubyte], %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_ann_d*, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.tree_complex = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, uint, %struct.tree_node*, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, uint, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, int, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, long, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { long }
	%struct.tree_decl_u1_a = type { uint }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_exp = type { %struct.tree_common, %struct.location_t*, int, %struct.tree_node*, [1 x %struct.tree_node*] }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_int_cst = type { %struct.tree_common, %struct.tree_int_cst_lowhi }
	%struct.tree_int_cst_lowhi = type { ulong, long }
	%struct.tree_list = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_live_info_d = type { %struct._var_map*, %struct.bitmap_head_def*, %struct.bitmap_head_def**, int, %struct.bitmap_head_def** }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_partition_associator_d = type { %struct.varray_head_tag*, %struct.varray_head_tag*, int*, int*, int, int, %struct._var_map* }
	%struct.tree_phi_node = type { %struct.tree_common, %struct.tree_node*, int, int, int, %struct.basic_block_def*, %struct.dataflow_d*, [1 x %struct.phi_arg_d] }
	%struct.tree_real_cst = type { %struct.tree_common, %struct.real_value* }
	%struct.tree_ssa_name = type { %struct.tree_common, %struct.tree_node*, uint, %struct.ptr_info_def*, %struct.tree_node*, sbyte* }
	%struct.tree_statement_list = type { %struct.tree_common, %struct.tree_statement_list_node*, %struct.tree_statement_list_node* }
	%struct.tree_statement_list_node = type { %struct.tree_statement_list_node*, %struct.tree_statement_list_node*, %struct.tree_node* }
	%struct.tree_stmt_iterator = type { %struct.tree_statement_list_node*, %struct.tree_node* }
	%struct.tree_string = type { %struct.tree_common, int, [1 x sbyte] }
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, uint, ushort, ubyte, ubyte, uint, %struct.tree_node*, %struct.tree_node*, %struct.tree_type_symtab, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, long, %struct.lang_type* }
	%struct.tree_type_symtab = type { int }
	%struct.tree_value_handle = type { %struct.tree_common, %struct.value_set*, uint }
	%struct.tree_vec = type { %struct.tree_common, int, [1 x %struct.tree_node*] }
	%struct.tree_vector = type { %struct.tree_common, %struct.tree_node* }
	%struct.use_operand_ptr = type { %struct.tree_node** }
	%struct.use_optype_d = type { uint, [1 x %struct.def_operand_ptr] }
	%struct.v_def_use_operand_type_t = type { %struct.tree_node*, %struct.tree_node* }
	%struct.v_may_def_optype_d = type { uint, [1 x %struct.v_def_use_operand_type_t] }
	%struct.v_must_def_optype_d = type { uint, [1 x %struct.v_def_use_operand_type_t] }
	%struct.value_set = type opaque
	%struct.var_ann_d = type { %struct.tree_ann_common_d, ubyte, ubyte, %struct.tree_node*, %struct.varray_head_tag*, uint, uint, uint, %struct.tree_node*, %struct.tree_node* }
	%struct.var_refs_queue = type { %struct.rtx_def*, uint, int, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x long] }
	%struct.varray_head_tag = type { uint, uint, uint, sbyte*, %struct.varray_data }
	%struct.vuse_optype_d = type { uint, [1 x %struct.tree_node*] }
%basic_block_info = external global %struct.varray_head_tag*		; <%struct.varray_head_tag**> [#uses=1]

implementation   ; Functions:


void %calculate_live_on_entry_cond_true3632(%struct.varray_head_tag* %stack3023.6, uint* %tmp3629, %struct.VEC_edge*** %tmp3397.out) {
newFuncRoot:
	br label %cond_true3632

bb3502.exitStub:		; preds = %cond_true3632
	store %struct.VEC_edge** %tmp3397, %struct.VEC_edge*** %tmp3397.out
	ret void

cond_true3632:		; preds = %newFuncRoot
	%tmp3378 = load uint* %tmp3629		; <uint> [#uses=1]
	%tmp3379 = add uint %tmp3378, 4294967295		; <uint> [#uses=1]
	%tmp3381 = getelementptr %struct.varray_head_tag* %stack3023.6, int 0, uint 4		; <%struct.varray_data*> [#uses=1]
	%tmp3382 = cast %struct.varray_data* %tmp3381 to [1 x int]*		; <[1 x int]*> [#uses=1]
	%tmp3383 = getelementptr [1 x int]* %tmp3382, int 0, uint %tmp3379		; <int*> [#uses=1]
	%tmp3384 = load int* %tmp3383		; <int> [#uses=1]
	%tmp3387 = load uint* %tmp3629		; <uint> [#uses=1]
	%tmp3388 = add uint %tmp3387, 4294967295		; <uint> [#uses=1]
	store uint %tmp3388, uint* %tmp3629
	%tmp3391 = load %struct.varray_head_tag** %basic_block_info		; <%struct.varray_head_tag*> [#uses=1]
	%tmp3393 = getelementptr %struct.varray_head_tag* %tmp3391, int 0, uint 4		; <%struct.varray_data*> [#uses=1]
	%tmp3394 = cast %struct.varray_data* %tmp3393 to [1 x %struct.basic_block_def*]*		; <[1 x %struct.basic_block_def*]*> [#uses=1]
	%tmp3395 = getelementptr [1 x %struct.basic_block_def*]* %tmp3394, int 0, int %tmp3384		; <%struct.basic_block_def**> [#uses=1]
	%tmp3396 = load %struct.basic_block_def** %tmp3395		; <%struct.basic_block_def*> [#uses=1]
	%tmp3397 = getelementptr %struct.basic_block_def* %tmp3396, int 0, uint 3		; <%struct.VEC_edge**> [#uses=1]
	br label %bb3502.exitStub
}
