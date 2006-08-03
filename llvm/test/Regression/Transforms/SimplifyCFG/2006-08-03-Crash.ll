; RUN: llvm-as < %s | opt -load-vn -gcse -simplifycfg -disable-output
; PR867

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8"
	%struct.CUMULATIVE_ARGS = type { int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.eh_status = type opaque
	%struct.emit_status = type { int, int, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, int, %struct.location_t, int, ubyte*, %struct.rtx_def** }
	%struct.expr_status = type { int, int, int, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, int, int, int, int, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, ubyte, int, long, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, int, %struct.var_refs_queue*, int, int, %struct.rtvec_def*, %struct.tree_node*, int, int, int, %struct.machine_function*, uint, uint, ubyte, ubyte, %struct.language_function*, %struct.rtx_def*, uint, int, int, int, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, ubyte, ubyte, ubyte }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { sbyte*, int }
	%struct.machine_function = type { int, uint, sbyte*, int, int }
	%struct.rtunion = type { int }
	%struct.rtvec_def = type { int, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { ushort, ubyte, ubyte, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, uint, %struct.tree_node*, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, uint, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, long, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { long }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, uint, ushort, ubyte, ubyte, uint, %struct.tree_node*, %struct.tree_node*, %struct.rtunion, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, long, %struct.lang_type* }
	%struct.u = type { [1 x long] }
	%struct.var_refs_queue = type { %struct.rtx_def*, uint, int, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { uint, uint, uint, sbyte*, %struct.u }
	%union.tree_ann_d = type opaque
%mode_class = external global [35 x ubyte]		; <[35 x ubyte]*> [#uses=3]

implementation   ; Functions:

void %fold_builtin_classify() {
entry:
	%tmp63 = load int* null		; <int> [#uses=1]
	switch int %tmp63, label %bb276 [
		 int 414, label %bb145
		 int 417, label %bb
	]

bb:		; preds = %entry
	ret void

bb145:		; preds = %entry
	%tmp146 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp148 = getelementptr %struct.tree_node* %tmp146, int 0, uint 0, uint 0, uint 1		; <%struct.tree_node**> [#uses=1]
	%tmp149 = load %struct.tree_node** %tmp148		; <%struct.tree_node*> [#uses=1]
	%tmp150 = cast %struct.tree_node* %tmp149 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp151 = getelementptr %struct.tree_type* %tmp150, int 0, uint 6		; <ushort*> [#uses=1]
	%tmp151 = cast ushort* %tmp151 to uint*		; <uint*> [#uses=1]
	%tmp152 = load uint* %tmp151		; <uint> [#uses=1]
	%tmp154 = shr uint %tmp152, ubyte 16		; <uint> [#uses=1]
	%tmp154.mask = and uint %tmp154, 127		; <uint> [#uses=1]
	%tmp155 = getelementptr [35 x ubyte]* %mode_class, int 0, uint %tmp154.mask		; <ubyte*> [#uses=1]
	%tmp156 = load ubyte* %tmp155		; <ubyte> [#uses=1]
	%tmp157 = seteq ubyte %tmp156, 4		; <bool> [#uses=1]
	br bool %tmp157, label %cond_next241, label %cond_true158

cond_true158:		; preds = %bb145
	%tmp172 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp174 = getelementptr %struct.tree_node* %tmp172, int 0, uint 0, uint 0, uint 1		; <%struct.tree_node**> [#uses=1]
	%tmp175 = load %struct.tree_node** %tmp174		; <%struct.tree_node*> [#uses=1]
	%tmp176 = cast %struct.tree_node* %tmp175 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp177 = getelementptr %struct.tree_type* %tmp176, int 0, uint 6		; <ushort*> [#uses=1]
	%tmp177 = cast ushort* %tmp177 to uint*		; <uint*> [#uses=1]
	%tmp178 = load uint* %tmp177		; <uint> [#uses=1]
	%tmp180 = shr uint %tmp178, ubyte 16		; <uint> [#uses=1]
	%tmp180.mask = and uint %tmp180, 127		; <uint> [#uses=1]
	%tmp181 = getelementptr [35 x ubyte]* %mode_class, int 0, uint %tmp180.mask		; <ubyte*> [#uses=1]
	%tmp182 = load ubyte* %tmp181		; <ubyte> [#uses=1]
	%tmp183 = seteq ubyte %tmp182, 8		; <bool> [#uses=1]
	br bool %tmp183, label %cond_next241, label %cond_true184

cond_true184:		; preds = %cond_true158
	%tmp185 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp187 = getelementptr %struct.tree_node* %tmp185, int 0, uint 0, uint 0, uint 1		; <%struct.tree_node**> [#uses=1]
	%tmp188 = load %struct.tree_node** %tmp187		; <%struct.tree_node*> [#uses=1]
	%tmp189 = cast %struct.tree_node* %tmp188 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp190 = getelementptr %struct.tree_type* %tmp189, int 0, uint 6		; <ushort*> [#uses=1]
	%tmp190 = cast ushort* %tmp190 to uint*		; <uint*> [#uses=1]
	%tmp191 = load uint* %tmp190		; <uint> [#uses=1]
	%tmp193 = shr uint %tmp191, ubyte 16		; <uint> [#uses=1]
	%tmp193.mask = and uint %tmp193, 127		; <uint> [#uses=1]
	%tmp194 = getelementptr [35 x ubyte]* %mode_class, int 0, uint %tmp193.mask		; <ubyte*> [#uses=1]
	%tmp195 = load ubyte* %tmp194		; <ubyte> [#uses=1]
	%tmp196 = seteq ubyte %tmp195, 4		; <bool> [#uses=1]
	br bool %tmp196, label %cond_next241, label %cond_true197

cond_true197:		; preds = %cond_true184
	ret void

cond_next241:		; preds = %cond_true184, %cond_true158, %bb145
	%tmp245 = load uint* null		; <uint> [#uses=0]
	ret void

bb276:		; preds = %entry
	ret void
}
