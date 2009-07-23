; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 | grep pinsrd | count 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 | grep pinsrb | count 1

define <4 x i32> @t1(i32 %s, <4 x i32> %tmp) nounwind {
        %tmp1 = insertelement <4 x i32> %tmp, i32 %s, i32 1
        ret <4 x i32> %tmp1
}

define <16 x i8> @t2(i8 %s, <16 x i8> %tmp) nounwind {
        %tmp1 = insertelement <16 x i8> %tmp, i8 %s, i32 1
        ret <16 x i8> %tmp1
}
