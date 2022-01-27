; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; CHECK: .weak .global .align 4 .u32 g
@g = common addrspace(1) global i32 zeroinitializer

define i32 @func0() {
  %val = load i32, i32 addrspace(1)* @g
  ret i32 %val
}
