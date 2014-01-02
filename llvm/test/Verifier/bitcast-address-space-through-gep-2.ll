; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: invalid cast opcode for cast from 'i32 addrspace(2)*' to 'i32 addrspace(3)*'

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-p3:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"


%struct.Foo1 = type { i32 addrspace(1)* }

@as2_array = addrspace(2) global [32 x i32] zeroinitializer

; gep -> legal bitcast (2 -> 3) -> gep -> illegal bitcast (3 -> 1)
@bitcast_after_gep_bitcast_gep =
         global %struct.Foo1 { i32 addrspace(1)* bitcast
                                    (i32 addrspace(3)* getelementptr
                                         (i32 addrspace(3)* bitcast
                                              (i32 addrspace(2)* getelementptr
                                                   ([32 x i32] addrspace(2)* @as2_array, i32 0, i32 8) to i32 addrspace(3)*), i32 3) to i32 addrspace(1)*) }

