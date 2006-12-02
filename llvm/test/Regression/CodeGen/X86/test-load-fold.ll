; RUN: llvm-upgrade < %s | llvm-as | llc
	%struct._obstack_chunk = type { sbyte*, %struct._obstack_chunk*, [4 x sbyte] }
	%struct.obstack = type { int, %struct._obstack_chunk*, sbyte*, sbyte*, sbyte*, int, int, %struct._obstack_chunk* (...)*, void (...)*, sbyte*, ubyte }
%stmt_obstack = external global %struct.obstack		; <%struct.obstack*> [#uses=1]

implementation   ; Functions:

void %expand_start_bindings() {
entry:
	br bool false, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%new_size.0.i = select bool false, int 0, int 0		; <int> [#uses=1]
	%tmp.i = load uint* cast (ubyte* getelementptr (%struct.obstack* %stmt_obstack, int 0, uint 10) to uint*)		; <uint> [#uses=1]
	%tmp.i = cast uint %tmp.i to ubyte		; <ubyte> [#uses=1]
	%tmp21.i = and ubyte %tmp.i, 1		; <ubyte> [#uses=1]
	%tmp22.i = seteq ubyte %tmp21.i, 0		; <bool> [#uses=1]
	br bool %tmp22.i, label %cond_false30.i, label %cond_true23.i

cond_true23.i:		; preds = %cond_true
	ret void

cond_false30.i:		; preds = %cond_true
	%tmp35.i = tail call %struct._obstack_chunk* null( int %new_size.0.i )		; <%struct._obstack_chunk*> [#uses=0]
	ret void

cond_next:		; preds = %entry
	ret void
}
