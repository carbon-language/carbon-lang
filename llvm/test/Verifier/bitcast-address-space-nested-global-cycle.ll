; RUN: not llvm-as -verify -disable-output %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

%struct.Self1 = type { %struct.Self1 addrspace(1)* }

@cycle1 = addrspace(1) constant %struct.Self1 { %struct.Self1 addrspace(1)* bitcast (%struct.Self1 addrspace(0)* @cycle0 to %struct.Self1 addrspace(1)*) }
@cycle0 = addrspace(0) constant %struct.Self1 { %struct.Self1 addrspace(1)* @cycle1 }
