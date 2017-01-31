; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Trivial optimization of generic addressing

; CHECK-LABEL: @load_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(1)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(1)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_global_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(1)*
  %tmp1 = load float, float addrspace(1)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_constant_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(2)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(2)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_constant_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(2)*
  %tmp1 = load float, float addrspace(2)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_group_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(3)*
; CHECK-NEXT: %tmp1 = load float, float addrspace(3)* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_group_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(3)*
  %tmp1 = load float, float addrspace(3)* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @load_private_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float*
; CHECK-NEXT: %tmp1 = load float, float* %tmp0
; CHECK-NEXT: ret float %tmp1
define float @load_private_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float*
  %tmp1 = load float, float* %tmp0
  ret float %tmp1
}

; CHECK-LABEL: @store_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(1)*
; CHECK-NEXT: store float 0.000000e+00, float addrspace(1)* %tmp0
define void @store_global_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(1)*
  store float 0.0, float addrspace(1)* %tmp0
  ret void
}

; CHECK-LABEL: @store_group_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(3)*
; CHECK-NEXT: store float 0.000000e+00, float addrspace(3)* %tmp0
define void @store_group_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float addrspace(3)*
  store float 0.0, float addrspace(3)* %tmp0
  ret void
}

; CHECK-LABEL: @store_private_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float*
; CHECK-NEXT: store float 0.000000e+00, float* %tmp0
define void @store_private_from_flat(float addrspace(4)* %generic_scalar) #0 {
  %tmp0 = addrspacecast float addrspace(4)* %generic_scalar to float*
  store float 0.0, float* %tmp0
  ret void
}

; optimized to global load/store.
; CHECK-LABEL: @load_store_global(
; CHECK-NEXT: %val = load i32, i32 addrspace(1)* %input, align 4
; CHECK-NEXT: store i32 %val, i32 addrspace(1)* %output, align 4
; CHECK-NEXT: ret void
define void @load_store_global(i32 addrspace(1)* nocapture %input, i32 addrspace(1)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(1)* %input to i32 addrspace(4)*
  %tmp1 = addrspacecast i32 addrspace(1)* %output to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %tmp0, align 4
  store i32 %val, i32 addrspace(4)* %tmp1, align 4
  ret void
}

; Optimized to group load/store.
; CHECK-LABEL: @load_store_group(
; CHECK-NEXT: %val = load i32, i32 addrspace(3)* %input, align 4
; CHECK-NEXT: store i32 %val, i32 addrspace(3)* %output, align 4
; CHECK-NEXT: ret void
define void @load_store_group(i32 addrspace(3)* nocapture %input, i32 addrspace(3)* nocapture %output) #0 {
  %tmp0 = addrspacecast i32 addrspace(3)* %input to i32 addrspace(4)*
  %tmp1 = addrspacecast i32 addrspace(3)* %output to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %tmp0, align 4
  store i32 %val, i32 addrspace(4)* %tmp1, align 4
  ret void
}

; Optimized to private load/store.
; CHECK-LABEL: @load_store_private(
; CHECK-NEXT: %val = load i32, i32* %input, align 4
; CHECK-NEXT: store i32 %val, i32* %output, align 4
; CHECK-NEXT: ret void
define void @load_store_private(i32* nocapture %input, i32* nocapture %output) #0 {
  %tmp0 = addrspacecast i32* %input to i32 addrspace(4)*
  %tmp1 = addrspacecast i32* %output to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %tmp0, align 4
  store i32 %val, i32 addrspace(4)* %tmp1, align 4
  ret void
}

; No optimization. flat load/store.
; CHECK-LABEL: @load_store_flat(
; CHECK-NEXT: %val = load i32, i32 addrspace(4)* %input, align 4
; CHECK-NEXT: store i32 %val, i32 addrspace(4)* %output, align 4
; CHECK-NEXT: ret void
define void @load_store_flat(i32 addrspace(4)* nocapture %input, i32 addrspace(4)* nocapture %output) #0 {
  %val = load i32, i32 addrspace(4)* %input, align 4
  store i32 %val, i32 addrspace(4)* %output, align 4
  ret void
}

; CHECK-LABEL: @store_addrspacecast_ptr_value(
; CHECK: %cast = addrspacecast i32 addrspace(1)* %input to i32 addrspace(4)*
; CHECK-NEXT: store i32 addrspace(4)* %cast, i32 addrspace(4)* addrspace(1)* %output, align 4
define void @store_addrspacecast_ptr_value(i32 addrspace(1)* nocapture %input, i32 addrspace(4)* addrspace(1)* nocapture %output) #0 {
  %cast = addrspacecast i32 addrspace(1)* %input to i32 addrspace(4)*
  store i32 addrspace(4)* %cast, i32 addrspace(4)* addrspace(1)* %output, align 4
  ret void
}

attributes #0 = { nounwind }
