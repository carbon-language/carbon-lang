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

