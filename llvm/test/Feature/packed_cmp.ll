; This test checks to make sure that NE and EQ comparisons of
; vector types work.
; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; XFAIL: *

%ivec_type = type <4 x i8> 
%ivec1  = constant %ivec_type < i8 1, i8 1, i8 1, i8 1 >
%ivec2  = constant %ivec_type < i8 0, i8 0, i8 0, i8 0 >

%fvec_type = type <4 x float>
%fvec1 = constant %fvec_type <float 1.0, float 1.0, float 1.0, float 1.0>
%fvec2 = constant %fvec_type <float 0.0, float 0.0, float 0.0, float 0.0>


define bool %ivectest1() {
    %v1 = load %ivec_type* getelementptr(%ivec_type* %ivec1, i32 0)
    %v2 = load %ivec_type* getelementptr(%ivec_type* %ivec2, i32 0)
    %res = icmp ne %ivec_type %v1, %v2
    ret bool %res
}

define bool %ivectest2() {
    %v1 = load %ivec_type* getelementptr(%ivec_type* %ivec1, i32 0)
    %v2 = load %ivec_type* getelementptr(%ivec_type* %ivec2, i32 0)
    %res = icmp eq %ivec_type %v1, %v2
    ret bool %res
}

define bool %fvectest1() {
    %v1 = load %fvec_type* %fvec1
    %v2 = load %fvec_type* %fvec2
    %res = fcmp one %fvec_type %v1, %v2
    ret bool %res
}

define bool %fvectest2() {
    %v1 = load %fvec_type* %fvec1
    %v2 = load %fvec_type* %fvec2
    %res = fcmp oeq %fvec_type %v1, %v2
    ret bool %res
}

define bool %fvectest3() {
    %v1 = load %fvec_type* %fvec1
    %v2 = load %fvec_type* %fvec2
    %res = fcmp une %fvec_type %v1, %v2
    ret bool %res
}

define bool %fvectest4() {
    %v1 = load %fvec_type* %fvec1
    %v2 = load %fvec_type* %fvec2
    %res = fcmp ueq %fvec_type %v1, %v2
    ret bool %res
}
