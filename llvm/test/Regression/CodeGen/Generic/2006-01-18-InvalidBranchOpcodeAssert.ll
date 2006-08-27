; RUN: llvm-as < %s | llc
; This crashed the PPC backend.

void %test() {
	%tmp125 = call bool %llvm.isunordered.f64( double 0.000000e+00, double 0.000000e+00 )		; <bool> [#uses=1]
	br bool %tmp125, label %bb154, label %cond_false133

cond_false133:		; preds = %entry
	ret void

bb154:		; preds = %entry
	%tmp164 = seteq uint 0, 0		; <bool> [#uses=0]
	ret void
}

declare bool %llvm.isunordered.f64(double, double)
