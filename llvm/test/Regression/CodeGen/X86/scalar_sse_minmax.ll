; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse1,+sse2 | grep mins | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse1,+sse2 | grep maxs | wc -l | grep 2

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



