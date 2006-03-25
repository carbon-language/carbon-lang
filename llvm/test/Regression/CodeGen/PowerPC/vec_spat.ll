; Test that vectors are scalarized/lowered correctly.
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vspltw | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g3 | grep stfs | wc -l | grep 4
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vsplti | wc -l | grep 2

%f4 = type <4 x float>
%i4 = type <4 x int>

implementation

void %splat(%f4* %P, %f4* %Q, float %X) {
        %tmp = insertelement %f4 undef, float %X, uint 0
        %tmp2 = insertelement %f4 %tmp, float %X, uint 1
        %tmp4 = insertelement %f4 %tmp2, float %X, uint 2
        %tmp6 = insertelement %f4 %tmp4, float %X, uint 3
	%q = load %f4* %Q
	%R = add %f4 %q, %tmp6
        store %f4 %R, %f4* %P
        ret void
}

void %splat_i4(%i4* %P, %i4* %Q, int %X) {
        %tmp = insertelement %i4 undef, int %X, uint 0
        %tmp2 = insertelement %i4 %tmp, int %X, uint 1
        %tmp4 = insertelement %i4 %tmp2, int %X, uint 2
        %tmp6 = insertelement %i4 %tmp4, int %X, uint 3
	%q = load %i4* %Q
	%R = add %i4 %q, %tmp6
        store %i4 %R, %i4* %P
        ret void
}

void %splat_imm_i32(%i4* %P, %i4* %Q, int %X) {
	%q = load %i4* %Q
	%R = add %i4 %q, <int -1, int -1, int -1, int -1>
        store %i4 %R, %i4* %P
        ret void
}

void %splat_imm_i16(%i4* %P, %i4* %Q, int %X) {
	%q = load %i4* %Q
	%R = add %i4 %q, <int 65537, int 65537, int 65537, int 65537>
        store %i4 %R, %i4* %P
        ret void
}

