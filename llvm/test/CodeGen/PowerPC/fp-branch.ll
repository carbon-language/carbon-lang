; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep fcmp | wc -l | grep 1

declare bool %llvm.isunordered.f64(double, double)

bool %intcoord_cond_next55(double %tmp48.reload) {
newFuncRoot:
	br label %cond_next55

bb72.exitStub:		; preds = %cond_next55
	ret bool true

cond_next62.exitStub:		; preds = %cond_next55
	ret bool false

cond_next55:		; preds = %newFuncRoot
	%tmp57 = setge double %tmp48.reload, 1.000000e+00		; <bool> [#uses=1]
	%tmp58 = tail call bool %llvm.isunordered.f64( double %tmp48.reload, double 1.000000e+00 )		; <bool> [#uses=1]
	%tmp59 = or bool %tmp57, %tmp58		; <bool> [#uses=1]
	br bool %tmp59, label %bb72.exitStub, label %cond_next62.exitStub
}
