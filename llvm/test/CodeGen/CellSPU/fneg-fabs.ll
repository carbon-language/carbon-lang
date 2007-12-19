; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep fsmbi   %t1.s | count 3 &&
; RUN: grep 32768   %t1.s | count 2 &&
; RUN: grep xor     %t1.s | count 4 &&
; RUN: grep and     %t1.s | count 5 &&
; RUN: grep andbi   %t1.s | count 3

define double @fneg_dp(double %X) {
	%Y = sub double -0.000000e+00, %X
	ret double %Y
}

define <2 x double> @fneg_dp_vec(<2 x double> %X) {
	%Y = sub <2 x double> < double -0.0000e+00, double -0.0000e+00 >, %X
	ret <2 x double> %Y
}

define float @fneg_sp(float %X) {
	%Y = sub float -0.000000e+00, %X
	ret float %Y
}

define <4 x float> @fneg_sp_vec(<4 x float> %X) {
	%Y = sub <4 x float> <float -0.000000e+00, float -0.000000e+00,
                              float -0.000000e+00, float -0.000000e+00>, %X
	ret <4 x float> %Y
}

declare double @fabs(double)

declare float @fabsf(float)

define double @fabs_dp(double %X) {
	%Y = call double @fabs( double %X )		; <double> [#uses=1]
	ret double %Y
}

define float @fabs_sp(float %X) {
	%Y = call float @fabsf( float %X )		; <float> [#uses=1]
	ret float %Y
}
