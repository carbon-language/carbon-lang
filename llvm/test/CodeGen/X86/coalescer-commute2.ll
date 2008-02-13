; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -mattr=+sse2 -coalescer-commute-instrs | grep movsd | count 4

define i32 @main(i32 %argc, i8** %argv) nounwind  {
entry:
	br label %bb145.us.i.i

bb145.us.i.i:		; preds = %bb145.us.i.i, %entry
	%seed.3.reg2mem.0.us.i.i = phi double [ 0.000000e+00, %entry ], [ %tmp9.i.us.i.i, %bb145.us.i.i ]		; <double> [#uses=1]
	%tmp2.i13.us.i.i = mul double %seed.3.reg2mem.0.us.i.i, 1.680700e+04		; <double> [#uses=1]
	%tmp3.i.us.i.i = add double %tmp2.i13.us.i.i, 1.000000e+00		; <double> [#uses=1]
	%tmp6.i15.us.i.i = call double @floor( double 0.000000e+00 ) nounwind readnone 		; <double> [#uses=1]
	%tmp7.i16.us.i.i = mul double %tmp6.i15.us.i.i, 0xC1DFFFFFFFC00000		; <double> [#uses=1]
	%tmp9.i.us.i.i = add double %tmp7.i16.us.i.i, %tmp3.i.us.i.i		; <double> [#uses=2]
	%tmp5.i12.us.i.i = mul double %tmp9.i.us.i.i, 2.000000e+00		; <double> [#uses=1]
	%tmp6.i.us.i.i = fdiv double %tmp5.i12.us.i.i, 0x41DFFFFFFFC00000		; <double> [#uses=1]
	%tmp8.i.us.i.i = add double %tmp6.i.us.i.i, -1.000000e+00		; <double> [#uses=1]
	store double %tmp8.i.us.i.i, double* null, align 8
	br label %bb145.us.i.i
}

declare double @floor(double) nounwind readnone 
