; RUN: llvm-as < %s | llc -march=sparc

void %execute_list() {
	%tmp.33.i = div float 0.000000e+00, 0.000000e+00		; <float> [#uses=1]
	%tmp.37.i = mul float 0.000000e+00, %tmp.33.i		; <float> [#uses=1]
	%tmp.42.i = add float %tmp.37.i, 0.000000e+00		; <float> [#uses=1]
	call void %gl_EvalCoord1f( float %tmp.42.i )
	ret void
}

declare void %gl_EvalCoord1f( float)

