; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep vector_shuffle

%T = type <4 x float>

implementation

%T %test1(%T %v1) {
  %v2 = shufflevector %T %v1, %T undef, <4 x uint> <uint 0, uint 1, uint 2, uint 3>
  ret %T %v2
}

%T %test2(%T %v1) {
  %v2 = shufflevector %T %v1, %T %v1, <4 x uint> <uint 0, uint 5, uint 2, uint 7>
  ret %T %v2
}

float %test3(%T %A, %T %B, float %f) {
        %C = insertelement %T %A, float %f, uint 0
        %D = shufflevector %T %C, %T %B, <4 x uint> <uint 5, uint 0, uint 2, uint 7>
        %E = extractelement %T %D, uint 1
        ret float %E
}

int %test4(<4 x int> %X) {
        %tmp152.i53899.i = shufflevector <4 x int> %X, <4 x int> undef, <4 x uint> zeroinitializer
        %tmp34 = extractelement <4 x int> %tmp152.i53899.i, uint 0
        ret int %tmp34
}

int %test5(<4 x int> %X) {
        %tmp152.i53899.i = shufflevector <4 x int> %X, <4 x int> undef, <4 x uint> <uint 3, uint 2, uint undef, uint undef>
        %tmp34 = extractelement <4 x int> %tmp152.i53899.i, uint 0
        ret int %tmp34
}

