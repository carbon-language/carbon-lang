; Test that vectors are scalarized/lowered correctly.
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vspltw | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g3 | grep stfs | wc -l | grep 4 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vsplti | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vsplth | wc -l | grep 1

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

void %splat_h(short %tmp, <16 x ubyte>* %dst) {
        %tmp = insertelement <8 x short> undef, short %tmp, uint 0
        %tmp72 = insertelement <8 x short> %tmp, short %tmp, uint 1
        %tmp73 = insertelement <8 x short> %tmp72, short %tmp, uint 2
        %tmp74 = insertelement <8 x short> %tmp73, short %tmp, uint 3
        %tmp75 = insertelement <8 x short> %tmp74, short %tmp, uint 4
        %tmp76 = insertelement <8 x short> %tmp75, short %tmp, uint 5
        %tmp77 = insertelement <8 x short> %tmp76, short %tmp, uint 6
        %tmp78 = insertelement <8 x short> %tmp77, short %tmp, uint 7
        %tmp78 = cast <8 x short> %tmp78 to <16 x ubyte>
        store <16 x ubyte> %tmp78, <16 x ubyte>* %dst
	ret void
}

void %spltish(<16 x ubyte>* %A, <16 x ubyte>* %B) {
	; Gets converted to 16 x ubyte 
        %tmp = load <16 x ubyte>* %B            
        %tmp = cast <16 x ubyte> %tmp to <16 x sbyte>           
        %tmp4 = sub <16 x sbyte> %tmp, cast (<8 x short> < short 15, short 15, short 15, short 15, short 15, short 15, short 15, short 15 > to <16 x sbyte>)            
        %tmp4 = cast <16 x sbyte> %tmp4 to <16 x ubyte>         
        store <16 x ubyte> %tmp4, <16 x ubyte>* %A
        ret void
}

