; RUN: llc < %s -march=ptx | FileCheck %s

;CHECK: .extern .global .f32 array[];
@array = external global [10 x float]

;CHECK: .extern .const .f32 array_constant[];
@array_constant = external addrspace(1) constant [10 x float]

;CHECK: .extern .local .f32 array_local[];
@array_local = external addrspace(2) global [10 x float]

;CHECK: .extern .shared .f32 array_shared[];
@array_shared = external addrspace(4) global [10 x float]

define ptx_device float @t1(float* %p) {
entry:
;CHECK: ld.global.f32 f0, [r1];
;CHECK-NEXT: ret;
  %x = load float* %p
  ret float %x
}

define ptx_device float @t2(float* %p) {
entry:
;CHECK: ld.global.f32 f0, [r1+4];
;CHECK-NEXT: ret;
  %i = getelementptr float* %p, i32 1
  %x = load float* %i
  ret float %x
}

define ptx_device float @t3(float* %p, i32 %q) {
entry:
;CHECK: shl.b32 r0, r2, 2;
;CHECK-NEXT: add.s32 r0, r1, r0;
;CHECK-NEXT: ld.global.f32 f0, [r0];
;CHECK-NEXT: ret;
  %i = getelementptr float* %p, i32 %q
  %x = load float* %i
  ret float %x
}

define ptx_device float @t4_global() {
entry:
;CHECK: ld.global.f32 f0, [array];
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float]* @array, i32 0, i32 0
  %x = load float* %i
  ret float %x
}

define ptx_device float @t4_const() {
entry:
;CHECK: ld.const.f32 f0, [array_constant];
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float] addrspace(1)* @array_constant, i32 0, i32 0
  %x = load float addrspace(1)* %i
  ret float %x
}

define ptx_device float @t4_local() {
entry:
;CHECK: ld.local.f32 f0, [array_local];
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float] addrspace(2)* @array_local, i32 0, i32 0
  %x = load float addrspace(2)* %i
  ret float %x
}

define ptx_device float @t4_shared() {
entry:
;CHECK: ld.shared.f32 f0, [array_shared];
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float] addrspace(4)* @array_shared, i32 0, i32 0
  %x = load float addrspace(4)* %i
  ret float %x
}

define ptx_device float @t5() {
entry:
;CHECK: ld.global.f32 f0, [array+4];
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float]* @array, i32 0, i32 1
  %x = load float* %i
  ret float %x
}
