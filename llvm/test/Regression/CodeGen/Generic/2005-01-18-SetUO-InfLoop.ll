; RUN: llvm-as < %s | llc

void %intersect_pixel() {
entry:
	%tmp125 = call bool %llvm.isunordered.f64( double 0.000000e+00, double 0.000000e+00 )		; <bool> [#uses=1]
	%tmp126 = or bool %tmp125, false		; <bool> [#uses=1]
	%tmp126.not = xor bool %tmp126, true		; <bool> [#uses=1]
	%brmerge1 = or bool %tmp126.not, false		; <bool> [#uses=1]
	br bool %brmerge1, label %bb154, label %cond_false133

cond_false133:		; preds = %entry
	ret void

bb154:		; preds = %entry
	%tmp164 = seteq uint 0, 0		; <bool> [#uses=0]
	ret void
}

declare bool %llvm.isunordered.f64(double, double)
