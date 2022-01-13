; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s

; CHECK:              ---
; CHECK:      amdhsa.kernels:

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK:          .name:           test0
; CHECK:          .symbol:         test0.kd
define amdgpu_kernel void @test0(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK:          .name:           test8
; CHECK:          .symbol:         test8.kd
define amdgpu_kernel void @test8(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK:          .name:           test16
; CHECK:          .symbol:         test16.kd
define amdgpu_kernel void @test16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #1 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK:          .name:           test24
; CHECK:          .symbol:         test24.kd
define amdgpu_kernel void @test24(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #2 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK:          .name:           test32
; CHECK:          .symbol:         test32.kd
define amdgpu_kernel void @test32(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #3 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK:          .name:           test48
; CHECK:          .symbol:         test48.kd
define amdgpu_kernel void @test48(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #4 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .name:           test56
; CHECK:          .symbol:         test56.kd
define amdgpu_kernel void @test56(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #5 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0

; We don't have a use of llvm.amdgcn.implicitarg.ptr, so optnone to
; avoid optimizing out the implicit argument allocation.
attributes #0 = { optnone noinline "amdgpu-implicitarg-num-bytes"="8" }
attributes #1 = { optnone noinline "amdgpu-implicitarg-num-bytes"="16" }
attributes #2 = { optnone noinline "amdgpu-implicitarg-num-bytes"="24" }
attributes #3 = { optnone noinline "amdgpu-implicitarg-num-bytes"="32" }
attributes #4 = { optnone noinline "amdgpu-implicitarg-num-bytes"="48" }
attributes #5 = { optnone noinline "amdgpu-implicitarg-num-bytes"="56" }
