; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s


; CHECK: .visible .global .align 4 .u32 device_g;
@device_g = addrspace(1) global i32 zeroinitializer
; CHECK: .visible .global .attribute(.managed) .align 4 .u32 managed_g;
@managed_g = addrspace(1) global i32 zeroinitializer


!nvvm.annotations = !{!0}
!0 = metadata !{i32 addrspace(1)* @managed_g, metadata !"managed", i32 1}
