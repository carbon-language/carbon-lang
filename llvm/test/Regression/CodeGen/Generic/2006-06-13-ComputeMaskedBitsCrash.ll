; RUN: llvm-as < %s | llc -fast

	%struct.cl_perfunc_opts = type { ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, int, int, int, int, int, int, int }
%cl_pf_opts = external global %struct.cl_perfunc_opts		; <%struct.cl_perfunc_opts*> [#uses=2]

implementation   ; Functions:

void %set_flags_from_O() {
entry:
	%tmp22 = setgt int 0, 0		; <bool> [#uses=1]
	br bool %tmp22, label %cond_true23, label %cond_next159

cond_true23:		; preds = %entry
	%tmp138 = getelementptr %struct.cl_perfunc_opts* %cl_pf_opts, int 0, uint 8		; <ubyte*> [#uses=1]
	%tmp138 = cast ubyte* %tmp138 to uint*		; <uint*> [#uses=2]
	%tmp139 = load uint* %tmp138		; <uint> [#uses=1]
	%tmp140 = shl uint 1, ubyte 27		; <uint> [#uses=1]
	%tmp141 = and uint %tmp140, 134217728		; <uint> [#uses=1]
	%tmp142 = and uint %tmp139, 4160749567		; <uint> [#uses=1]
	%tmp143 = or uint %tmp142, %tmp141		; <uint> [#uses=1]
	store uint %tmp143, uint* %tmp138
	%tmp144 = getelementptr %struct.cl_perfunc_opts* %cl_pf_opts, int 0, uint 8		; <ubyte*> [#uses=1]
	%tmp144 = cast ubyte* %tmp144 to uint*		; <uint*> [#uses=1]
	%tmp145 = load uint* %tmp144		; <uint> [#uses=1]
	%tmp146 = shl uint %tmp145, ubyte 22		; <uint> [#uses=1]
	%tmp147 = shr uint %tmp146, ubyte 31		; <uint> [#uses=1]
	%tmp147 = cast uint %tmp147 to ubyte		; <ubyte> [#uses=1]
	%tmp148 = seteq ubyte %tmp147, 0		; <bool> [#uses=1]
	br bool %tmp148, label %cond_true149, label %cond_next159

cond_true149:		; preds = %cond_true23
	%tmp150 = cast ubyte* null to uint*		; <uint*> [#uses=0]
	ret void

cond_next159:		; preds = %cond_true23, %entry
	ret void
}
