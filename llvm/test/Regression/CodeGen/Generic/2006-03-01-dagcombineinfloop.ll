; RUN: llvm-as < %s | llc
; Infinite loop in the dag combiner, reduced from 176.gcc.

	%struct._obstack_chunk = type { sbyte*, %struct._obstack_chunk*, [4 x sbyte] }
	%struct.anon = type { int }
	%struct.lang_decl = type opaque
	%struct.lang_type = type { int, [1 x %struct.tree_node*] }
	%struct.obstack = type { int, %struct._obstack_chunk*, sbyte*, sbyte*, sbyte*, int, int, %struct._obstack_chunk* (...)*, void (...)*, sbyte*, ubyte }
	%struct.rtx_def = type { ushort, ubyte, ubyte, [1 x %struct.anon] }
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, ubyte, ubyte, ubyte, ubyte }
	%struct.tree_decl = type { [12 x sbyte], sbyte*, int, %struct.tree_node*, uint, ubyte, ubyte, ubyte, ubyte, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.anon, { %struct.rtx_def* }, %struct.tree_node*, %struct.lang_decl* }
	%struct.tree_list = type { [12 x sbyte], %struct.tree_node*, %struct.tree_node* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_type = type { [12 x sbyte], %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, uint, ubyte, ubyte, ubyte, ubyte, uint, %struct.tree_node*, %struct.tree_node*, %struct.anon, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.obstack*, %struct.lang_type* }
%void_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]
%char_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]
%short_integer_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]
%short_unsigned_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]
%float_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]
%signed_char_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]
%unsigned_char_type_node = external global %struct.tree_node*		; <%struct.tree_node**> [#uses=1]

implementation   ; Functions:

fastcc int %self_promoting_args_p(%struct.tree_node* %parms) {
entry:
	%tmp915 = seteq %struct.tree_node* %parms, null		; <bool> [#uses=1]
	br bool %tmp915, label %return, label %cond_true92.preheader

cond_true:		; preds = %cond_true92
	%tmp9.not = setne %struct.tree_node* %tmp2, %tmp7		; <bool> [#uses=1]
	%tmp14 = seteq %struct.tree_node* %tmp2, null		; <bool> [#uses=1]
	%bothcond = or bool %tmp9.not, %tmp14		; <bool> [#uses=1]
	br bool %bothcond, label %return, label %cond_next18

cond_next12:		; preds = %cond_true92
	%tmp14.old = seteq %struct.tree_node* %tmp2, null		; <bool> [#uses=1]
	br bool %tmp14.old, label %return, label %cond_next18

cond_next18:		; preds = %cond_next12, %cond_true
	%tmp20 = cast %struct.tree_node* %tmp2 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp21 = getelementptr %struct.tree_type* %tmp20, int 0, uint 17		; <%struct.tree_node**> [#uses=1]
	%tmp22 = load %struct.tree_node** %tmp21		; <%struct.tree_node*> [#uses=6]
	%tmp24 = seteq %struct.tree_node* %tmp22, %tmp23		; <bool> [#uses=1]
	br bool %tmp24, label %return, label %cond_next28

cond_next28:		; preds = %cond_next18
	%tmp30 = cast %struct.tree_node* %tmp2 to %struct.tree_common*		; <%struct.tree_common*> [#uses=1]
	%tmp = getelementptr %struct.tree_common* %tmp30, int 0, uint 2		; <ubyte*> [#uses=1]
	%tmp = cast ubyte* %tmp to uint*		; <uint*> [#uses=1]
	%tmp = load uint* %tmp		; <uint> [#uses=1]
	%tmp32 = cast uint %tmp to ubyte		; <ubyte> [#uses=1]
	%tmp33 = seteq ubyte %tmp32, 7		; <bool> [#uses=1]
	br bool %tmp33, label %cond_true34, label %cond_next84

cond_true34:		; preds = %cond_next28
	%tmp40 = seteq %struct.tree_node* %tmp22, %tmp39		; <bool> [#uses=1]
	%tmp49 = seteq %struct.tree_node* %tmp22, %tmp48		; <bool> [#uses=1]
	%bothcond6 = or bool %tmp40, %tmp49		; <bool> [#uses=1]
	%tmp58 = seteq %struct.tree_node* %tmp22, %tmp57		; <bool> [#uses=1]
	%bothcond7 = or bool %bothcond6, %tmp58		; <bool> [#uses=1]
	%tmp67 = seteq %struct.tree_node* %tmp22, %tmp66		; <bool> [#uses=1]
	%bothcond8 = or bool %bothcond7, %tmp67		; <bool> [#uses=1]
	%tmp76 = seteq %struct.tree_node* %tmp22, %tmp75		; <bool> [#uses=1]
	%bothcond9 = or bool %bothcond8, %tmp76		; <bool> [#uses=2]
	%brmerge = or bool %bothcond9, %tmp		; <bool> [#uses=1]
	%bothcond9 = cast bool %bothcond9 to int		; <int> [#uses=1]
	%.mux = xor int %bothcond9, 1		; <int> [#uses=1]
	br bool %brmerge, label %return, label %cond_true92

cond_next84:		; preds = %cond_next28
	br bool %tmp, label %return, label %cond_true92

cond_true92.preheader:		; preds = %entry
	%tmp7 = load %struct.tree_node** %void_type_node		; <%struct.tree_node*> [#uses=1]
	%tmp23 = load %struct.tree_node** %float_type_node		; <%struct.tree_node*> [#uses=1]
	%tmp39 = load %struct.tree_node** %char_type_node		; <%struct.tree_node*> [#uses=1]
	%tmp48 = load %struct.tree_node** %signed_char_type_node		; <%struct.tree_node*> [#uses=1]
	%tmp57 = load %struct.tree_node** %unsigned_char_type_node		; <%struct.tree_node*> [#uses=1]
	%tmp66 = load %struct.tree_node** %short_integer_type_node		; <%struct.tree_node*> [#uses=1]
	%tmp75 = load %struct.tree_node** %short_unsigned_type_node		; <%struct.tree_node*> [#uses=1]
	br label %cond_true92

cond_true92:		; preds = %cond_true92.preheader, %cond_next84, %cond_true34
	%t.0.0 = phi %struct.tree_node* [ %parms, %cond_true92.preheader ], [ %tmp6, %cond_true34 ], [ %tmp6, %cond_next84 ]		; <%struct.tree_node*> [#uses=2]
	%tmp = cast %struct.tree_node* %t.0.0 to %struct.tree_list*		; <%struct.tree_list*> [#uses=1]
	%tmp = getelementptr %struct.tree_list* %tmp, int 0, uint 2		; <%struct.tree_node**> [#uses=1]
	%tmp2 = load %struct.tree_node** %tmp		; <%struct.tree_node*> [#uses=5]
	%tmp4 = cast %struct.tree_node* %t.0.0 to %struct.tree_common*		; <%struct.tree_common*> [#uses=1]
	%tmp5 = getelementptr %struct.tree_common* %tmp4, int 0, uint 0		; <%struct.tree_node**> [#uses=1]
	%tmp6 = load %struct.tree_node** %tmp5		; <%struct.tree_node*> [#uses=3]
	%tmp = seteq %struct.tree_node* %tmp6, null		; <bool> [#uses=3]
	br bool %tmp, label %cond_true, label %cond_next12

return:		; preds = %cond_next84, %cond_true34, %cond_next18, %cond_next12, %cond_true, %entry
	%retval.0 = phi int [ 1, %entry ], [ 1, %cond_next84 ], [ %.mux, %cond_true34 ], [ 0, %cond_next18 ], [ 0, %cond_next12 ], [ 0, %cond_true ]		; <int> [#uses=1]
	ret int %retval.0
}
