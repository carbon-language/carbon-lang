; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

fastcc int %test(%struct.foo* %v, %struct.foo* %vi) {
	br bool false, label %ilog2.exit, label %cond_true.i

cond_true.i:		; preds = %entry
	ret int 0

ilog2.exit:		; preds = %entry
	%tmp24.i = load int* null		; <int> [#uses=1]
	%tmp13.i12.i = tail call double %ldexp( double 0.000000e+00, int 0 )		; <double> [#uses=1]
	%tmp13.i13.i = cast double %tmp13.i12.i to float		; <float> [#uses=1]
	%tmp11.i = load int* null		; <int> [#uses=1]
	%tmp11.i = cast int %tmp11.i to uint		; <uint> [#uses=1]
	%n.i = cast int %tmp24.i to uint		; <uint> [#uses=1]
	%tmp13.i7 = mul uint %tmp11.i, %n.i		; <uint> [#uses=1]
	%tmp.i8 = tail call sbyte* %calloc( uint %tmp13.i7, uint 4 )		; <sbyte*> [#uses=0]
	br bool false, label %bb224.preheader.i, label %bb.i

bb.i:		; preds = %ilog2.exit
	ret int 0

bb224.preheader.i:		; preds = %ilog2.exit
	%tmp165.i = cast float %tmp13.i13.i to double		; <double> [#uses=0]
	ret int 0
}

declare sbyte* %calloc(uint, uint)

declare double %ldexp(double, int)
