; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: invalid cast opcode for cast from 'i32 addrspace(2)*' to 'i32 addrspace(1)*'

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"


@data = addrspace(2) global i32 27

@illegal_alias_data = alias bitcast (i32 addrspace(2)* @data to i32 addrspace(1)*)
