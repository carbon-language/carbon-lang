; RUN: llc < %s -O0
	
%struct.cl_perfunc_opts = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32 }
@cl_pf_opts = external global %struct.cl_perfunc_opts		; <%struct.cl_perfunc_opts*> [#uses=2]

define void @set_flags_from_O() {
entry:
	%tmp22 = icmp sgt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp22, label %cond_true23, label %cond_next159

cond_true23:		; preds = %entry
	%tmp138 = getelementptr %struct.cl_perfunc_opts, %struct.cl_perfunc_opts* @cl_pf_opts, i32 0, i32 8		; <i8*> [#uses=1]
	%tmp138.upgrd.1 = bitcast i8* %tmp138 to i32*		; <i32*> [#uses=2]
	%tmp139 = load i32, i32* %tmp138.upgrd.1		; <i32> [#uses=1]
	%tmp140 = shl i32 1, 27		; <i32> [#uses=1]
	%tmp141 = and i32 %tmp140, 134217728		; <i32> [#uses=1]
	%tmp142 = and i32 %tmp139, -134217729		; <i32> [#uses=1]
	%tmp143 = or i32 %tmp142, %tmp141		; <i32> [#uses=1]
	store i32 %tmp143, i32* %tmp138.upgrd.1
	%tmp144 = getelementptr %struct.cl_perfunc_opts, %struct.cl_perfunc_opts* @cl_pf_opts, i32 0, i32 8		; <i8*> [#uses=1]
	%tmp144.upgrd.2 = bitcast i8* %tmp144 to i32*		; <i32*> [#uses=1]
	%tmp145 = load i32, i32* %tmp144.upgrd.2		; <i32> [#uses=1]
	%tmp146 = shl i32 %tmp145, 22		; <i32> [#uses=1]
	%tmp147 = lshr i32 %tmp146, 31		; <i32> [#uses=1]
	%tmp147.upgrd.3 = trunc i32 %tmp147 to i8		; <i8> [#uses=1]
	%tmp148 = icmp eq i8 %tmp147.upgrd.3, 0		; <i1> [#uses=1]
	br i1 %tmp148, label %cond_true149, label %cond_next159

cond_true149:		; preds = %cond_true23
	%tmp150 = bitcast i8* null to i32*		; <i32*> [#uses=0]
	ret void

cond_next159:		; preds = %cond_true23, %entry
	ret void
}
