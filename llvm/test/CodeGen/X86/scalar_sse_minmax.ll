; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse1,+sse2 | \
; RUN:   grep mins | wc -l | grep 3
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse1,+sse2 | \
; RUN:   grep maxs | wc -l | grep 2

declare bool %llvm.isunordered.f64( double %x, double %y )
declare bool %llvm.isunordered.f32( float %x, float %y )

implementation

float %min1(float %x, float %y) {
        %tmp = setlt float %x, %y               ; <bool> [#uses=1]
        %retval = select bool %tmp, float %x, float %y          ; <float> [#uses=1]
        ret float %retval
}
double %min2(double %x, double %y) {
        %tmp = setlt double %x, %y
        %retval = select bool %tmp, double %x, double %y
        ret double %retval
}

float %max1(float %x, float %y) {
        %tmp = setge float %x, %y               ; <bool> [#uses=1]
        %tmp2 = tail call bool %llvm.isunordered.f32( float %x, float %y )
        %tmp3 = or bool %tmp2, %tmp             ; <bool> [#uses=1]
        %retval = select bool %tmp3, float %x, float %y         
        ret float %retval
}

double %max2(double %x, double %y) {
        %tmp = setge double %x, %y               ; <bool> [#uses=1]
        %tmp2 = tail call bool %llvm.isunordered.f64( double %x, double %y )
        %tmp3 = or bool %tmp2, %tmp             ; <bool> [#uses=1]
        %retval = select bool %tmp3, double %x, double %y
        ret double %retval
}

<4 x float> %min3(float %tmp37) {
        %tmp375 = insertelement <4 x float> undef, float %tmp37, uint 0
        %tmp48 = tail call <4 x float> %llvm.x86.sse.min.ss( <4 x float> %tmp375, <4 x float> < float 6.553500e+04, float undef, float undef, float undef > )
	ret <4 x float> %tmp48
}

declare <4 x float> %llvm.x86.sse.min.ss(<4 x float>, <4 x float>)
