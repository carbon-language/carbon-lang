; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-any-address-space-out-arguments -amdgpu-rewrite-out-arguments < %s | FileCheck %s


; CHECK: %void_one_out_non_private_arg_i32_1_use = type { i32 }


; CHECK-LABEL: define private %void_one_out_non_private_arg_i32_1_use @void_one_out_non_private_arg_i32_1_use.body(i32 addrspace(1)* %val) #0 {
; CHECK-NEXT: ret %void_one_out_non_private_arg_i32_1_use zeroinitializer

; CHECK-LABEL: define void @void_one_out_non_private_arg_i32_1_use(i32 addrspace(1)*) #1 {
; CHECK-NEXT: %2 = call %void_one_out_non_private_arg_i32_1_use @void_one_out_non_private_arg_i32_1_use.body(i32 addrspace(1)* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_non_private_arg_i32_1_use %2, 0
; CHECK-NEXT: store i32 %3, i32 addrspace(1)* %0, align 4
; CHECK-NEXT: ret void
define void @void_one_out_non_private_arg_i32_1_use(i32 addrspace(1)* %val) #0 {
  store i32 0, i32 addrspace(1)* %val
  ret void
}

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes #1 = { alwaysinline nounwind }
attributes #0 = { nounwind }
