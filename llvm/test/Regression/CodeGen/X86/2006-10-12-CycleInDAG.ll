; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86
	%struct.function = type opaque
	%struct.lang_decl = type opaque
	%struct.location_t = type { sbyte*, int }
	%struct.rtx_def = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, ubyte, ubyte, ubyte, ubyte, ubyte }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, uint, %struct.tree_node*, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, uint, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, int, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, long, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { long }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%union.tree_ann_d = type opaque

void %check_format_arg() {
	br bool false, label %cond_next196, label %bb12.preheader

bb12.preheader:
	ret void

cond_next196:
	br bool false, label %cond_next330, label %cond_true304

cond_true304:
	ret void

cond_next330:
	br bool false, label %cond_next472, label %bb441

bb441:
	ret void

cond_next472:
	%tmp490 = load %struct.tree_node** null
	%tmp492 = getelementptr %struct.tree_node* %tmp490, int 0, uint 0, uint 0, uint 3
	%tmp492 = cast ubyte* %tmp492 to uint*
	%tmp493 = load uint* %tmp492
	%tmp495 = cast uint %tmp493 to ubyte
	%tmp496 = seteq ubyte %tmp495, 11
	%tmp496 = cast bool %tmp496 to sbyte
	store sbyte %tmp496, sbyte* null
	ret void
}
