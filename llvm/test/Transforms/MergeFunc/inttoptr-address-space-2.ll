; RUN: opt -mergefunc -S < %s | FileCheck %s
target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-n8:16:32-S128"

%.qux.2496 = type { i16, %.qux.2497 }
%.qux.2497 = type { i8, i16 }
%.qux.2585 = type { i16, i16, i8 addrspace(1)* }

@g2 = external addrspace(1) constant [9 x i8], align 1
@g3 = internal hidden addrspace(1) constant [1 x i8*] [i8* bitcast (i8 addrspace(1)* (%.qux.2585 addrspace(1)*)* @func35 to i8*)]


define internal hidden i16 @func10(%.qux.2496 addrspace(1)* nocapture %this) align 2 {
bb:
  %tmp = getelementptr inbounds %.qux.2496 addrspace(1)* %this, i32 0, i32 1, i32 1
  %tmp1 = load i16 addrspace(1)* %tmp, align 4
  ret i16 %tmp1
}

; Checks that this can be merged with an address space differently sized than 0
define internal hidden i8 addrspace(1)* @func35(%.qux.2585 addrspace(1)* nocapture %this) align 2 {
bb:
; CHECK-LABEL: @func35(
; CHECK: %[[V2:.+]] = bitcast %.qux.2585 addrspace(1)* %{{.*}} to %.qux.2496 addrspace(1)*
; CHECK: %[[V3:.+]] = tail call i16 @func10(%.qux.2496 addrspace(1)* %[[V2]])
; CHECK: %{{.*}} = inttoptr i16 %[[V3]] to i8 addrspace(1)*
  %tmp = getelementptr inbounds %.qux.2585 addrspace(1)* %this, i32 0, i32 2
  %tmp1 = load i8 addrspace(1)* addrspace(1)* %tmp, align 4
  ret i8 addrspace(1)* %tmp1
}
