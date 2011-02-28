; RUN: llc < %s -march=ptx | FileCheck %s

;CHECK: .extern .global .f32 array[];
@array = external global [10 x float]

;CHECK: .extern .const .f32 array_constant[];
@array_constant = external addrspace(1) constant [10 x float]

;CHECK: .extern .local .f32 array_local[];
@array_local = external addrspace(2) global [10 x float]

;CHECK: .extern .shared .f32 array_shared[];
@array_shared = external addrspace(4) global [10 x float]

define ptx_device void @t1(float* %p, float %x) {
entry:
;CHECK: st.global.f32 [r1], f1;
;CHECK-NEXT: ret;
  store float %x, float* %p
  ret void
}

define ptx_device void @t2(float* %p, float %x) {
entry:
;CHECK: st.global.f32 [r1+4], f1;
;CHECK-NEXT: ret;
  %i = getelementptr float* %p, i32 1
  store float %x, float* %i
  ret void
}

define ptx_device void @t3(float* %p, i32 %q, float %x) {
;CHECK: .reg .s32 r0;
entry:
;CHECK: shl.b32 r0, r2, 2;
;CHECK-NEXT: add.s32 r0, r1, r0;
;CHECK-NEXT: st.global.f32 [r0], f1;
;CHECK-NEXT: ret;
  %i = getelementptr float* %p, i32 %q
  store float %x, float* %i
  ret void
}

define ptx_device void @t4_global(float %x) {
entry:
;CHECK: st.global.f32 [array], f1;
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float]* @array, i32 0, i32 0
  store float %x, float* %i
  ret void
}

define ptx_device void @t4_local(float %x) {
entry:
;CHECK: st.local.f32 [array_local], f1;
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float] addrspace(2)* @array_local, i32 0, i32 0
  store float %x, float addrspace(2)* %i
  ret void
}

define ptx_device void @t4_shared(float %x) {
entry:
;CHECK: st.shared.f32 [array_shared], f1;
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float] addrspace(4)* @array_shared, i32 0, i32 0
  store float %x, float addrspace(4)* %i
  ret void
}

define ptx_device void @t5(float %x) {
entry:
;CHECK: st.global.f32 [array+4], f1;
;CHECK-NEXT: ret;
  %i = getelementptr [10 x float]* @array, i32 0, i32 1
  store float %x, float* %i
  ret void
}
