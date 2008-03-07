; RUN: llvm-as < %s | opt -inline -sretpromotion -disable-output
	%struct.Demand = type { double, double }
	%struct.branch = type { %struct.Demand, double, double, double, double, %struct.branch*, [12 x %struct.leaf*] }
	%struct.leaf = type { %struct.Demand, double, double }
@P = external global double		; <double*> [#uses=1]

define %struct.leaf* @build_leaf() nounwind  {
entry:
	unreachable
}

define void @Compute_Branch(%struct.Demand* sret  %agg.result, %struct.branch* %br, double %theta_R, double %theta_I, double %pi_R, double %pi_I) nounwind  {
entry:
	%a2 = alloca %struct.Demand		; <%struct.Demand*> [#uses=2]
	br i1 false, label %bb46, label %bb

bb:		; preds = %entry
	ret void

bb46:		; preds = %entry
	br label %bb72

bb49:		; preds = %bb72
	call void @Compute_Leaf( %struct.Demand* sret  %a2, %struct.leaf* null, double 0.000000e+00, double 0.000000e+00 ) nounwind 
	%tmp66 = getelementptr %struct.Demand* %a2, i32 0, i32 1		; <double*> [#uses=0]
	br label %bb72

bb72:		; preds = %bb49, %bb46
	br i1 false, label %bb49, label %bb77

bb77:		; preds = %bb72
	ret void
}

define void @Compute_Leaf(%struct.Demand* sret  %agg.result, %struct.leaf* %l, double %pi_R, double %pi_I) nounwind  {
entry:
	%tmp10 = load double* @P, align 8		; <double> [#uses=1]
	%tmp11 = fcmp olt double %tmp10, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp11, label %bb, label %bb13

bb:		; preds = %entry
	ret void

bb13:		; preds = %entry
	ret void
}
