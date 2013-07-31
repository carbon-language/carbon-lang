; RUN: not llvm-as -verify -disable-output %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

define <4 x i32 addrspace(1)*> @vector_illegal_bitcast_as_0_to_1(<4 x i32 addrspace(0)*> %p) {
   %cast = bitcast <4 x i32 addrspace(0)*> %p to <4 x i32 addrspace(1)*>
   ret <4 x i32 addrspace(1)*> %cast
}

