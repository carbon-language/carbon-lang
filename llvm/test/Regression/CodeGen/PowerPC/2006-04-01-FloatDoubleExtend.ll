; RUN: llvm-as < %s | llc -march=ppc32

double %CalcSpeed(float %tmp127) {
	%tmp145 = cast float %tmp127 to double		; <double> [#uses=1]
	%tmp150 = call double asm "frsqrte $0,$1", "=f,f"( double %tmp145 )		; <double> [#uses=0]
	ret double %tmp150
}
