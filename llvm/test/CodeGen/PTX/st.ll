; RUN: llc < %s -march=ptx32 | FileCheck %s

;CHECK: .extern .global .b8 array_i16[20];
@array_i16 = external global [10 x i16]

;CHECK: .extern .const .b8 array_constant_i16[20];
@array_constant_i16 = external addrspace(1) constant [10 x i16]

;CHECK: .extern .local .b8 array_local_i16[20];
@array_local_i16 = external addrspace(2) global [10 x i16]

;CHECK: .extern .shared .b8 array_shared_i16[20];
@array_shared_i16 = external addrspace(4) global [10 x i16]

;CHECK: .extern .global .b8 array_i32[40];
@array_i32 = external global [10 x i32]

;CHECK: .extern .const .b8 array_constant_i32[40];
@array_constant_i32 = external addrspace(1) constant [10 x i32]

;CHECK: .extern .local .b8 array_local_i32[40];
@array_local_i32 = external addrspace(2) global [10 x i32]

;CHECK: .extern .shared .b8 array_shared_i32[40];
@array_shared_i32 = external addrspace(4) global [10 x i32]

;CHECK: .extern .global .b8 array_i64[80];
@array_i64 = external global [10 x i64]

;CHECK: .extern .const .b8 array_constant_i64[80];
@array_constant_i64 = external addrspace(1) constant [10 x i64]

;CHECK: .extern .local .b8 array_local_i64[80];
@array_local_i64 = external addrspace(2) global [10 x i64]

;CHECK: .extern .shared .b8 array_shared_i64[80];
@array_shared_i64 = external addrspace(4) global [10 x i64]

;CHECK: .extern .global .b8 array_float[40];
@array_float = external global [10 x float]

;CHECK: .extern .const .b8 array_constant_float[40];
@array_constant_float = external addrspace(1) constant [10 x float]

;CHECK: .extern .local .b8 array_local_float[40];
@array_local_float = external addrspace(2) global [10 x float]

;CHECK: .extern .shared .b8 array_shared_float[40];
@array_shared_float = external addrspace(4) global [10 x float]

;CHECK: .extern .global .b8 array_double[80];
@array_double = external global [10 x double]

;CHECK: .extern .const .b8 array_constant_double[80];
@array_constant_double = external addrspace(1) constant [10 x double]

;CHECK: .extern .local .b8 array_local_double[80];
@array_local_double = external addrspace(2) global [10 x double]

;CHECK: .extern .shared .b8 array_shared_double[80];
@array_shared_double = external addrspace(4) global [10 x double]


define ptx_device void @t1_u16(i16* %p, i16 %x) {
entry:
;CHECK: st.global.u16 [%r{{[0-9]+}}], %rh{{[0-9]+}};
;CHECK: ret;
  store i16 %x, i16* %p
  ret void
}

define ptx_device void @t1_u32(i32* %p, i32 %x) {
entry:
;CHECK: st.global.u32 [%r{{[0-9]+}}], %r{{[0-9]+}};
;CHECK: ret;
  store i32 %x, i32* %p
  ret void
}

define ptx_device void @t1_u64(i64* %p, i64 %x) {
entry:
;CHECK: st.global.u64 [%r{{[0-9]+}}], %rd{{[0-9]+}};
;CHECK: ret;
  store i64 %x, i64* %p
  ret void
}

define ptx_device void @t1_f32(float* %p, float %x) {
entry:
;CHECK: st.global.f32 [%r{{[0-9]+}}], %f{{[0-9]+}};
;CHECK: ret;
  store float %x, float* %p
  ret void
}

define ptx_device void @t1_f64(double* %p, double %x) {
entry:
;CHECK: st.global.f64 [%r{{[0-9]+}}], %fd{{[0-9]+}};
;CHECK: ret;
  store double %x, double* %p
  ret void
}

define ptx_device void @t2_u16(i16* %p, i16 %x) {
entry:
;CHECK: st.global.u16 [%r{{[0-9]+}}+2], %rh{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr i16* %p, i32 1
  store i16 %x, i16* %i
  ret void
}

define ptx_device void @t2_u32(i32* %p, i32 %x) {
entry:
;CHECK: st.global.u32 [%r{{[0-9]+}}+4], %r{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr i32* %p, i32 1
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t2_u64(i64* %p, i64 %x) {
entry:
;CHECK: st.global.u64 [%r{{[0-9]+}}+8], %rd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr i64* %p, i32 1
  store i64 %x, i64* %i
  ret void
}

define ptx_device void @t2_f32(float* %p, float %x) {
entry:
;CHECK: st.global.f32 [%r{{[0-9]+}}+4], %f{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr float* %p, i32 1
  store float %x, float* %i
  ret void
}

define ptx_device void @t2_f64(double* %p, double %x) {
entry:
;CHECK: st.global.f64 [%r{{[0-9]+}}+8], %fd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr double* %p, i32 1
  store double %x, double* %i
  ret void
}

define ptx_device void @t3_u16(i16* %p, i32 %q, i16 %x) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 1;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: st.global.u16 [%r{{[0-9]+}}], %rh{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr i16* %p, i32 %q
  store i16 %x, i16* %i
  ret void
}

define ptx_device void @t3_u32(i32* %p, i32 %q, i32 %x) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 2;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: st.global.u32 [%r{{[0-9]+}}], %r{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr i32* %p, i32 %q
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t3_u64(i64* %p, i32 %q, i64 %x) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 3;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: st.global.u64 [%r{{[0-9]+}}], %rd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr i64* %p, i32 %q
  store i64 %x, i64* %i
  ret void
}

define ptx_device void @t3_f32(float* %p, i32 %q, float %x) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 2;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: st.global.f32 [%r{{[0-9]+}}], %f{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr float* %p, i32 %q
  store float %x, float* %i
  ret void
}

define ptx_device void @t3_f64(double* %p, i32 %q, double %x) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 3;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: st.global.f64 [%r{{[0-9]+}}], %fd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr double* %p, i32 %q
  store double %x, double* %i
  ret void
}

define ptx_device void @t4_global_u16(i16 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i16;
;CHECK: st.global.u16 [%r[[R0]]], %rh{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i16]* @array_i16, i16 0, i16 0
  store i16 %x, i16* %i
  ret void
}

define ptx_device void @t4_global_u32(i32 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i32;
;CHECK: st.global.u32 [%r[[R0]]], %r{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i32]* @array_i32, i32 0, i32 0
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t4_global_u64(i64 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i64;
;CHECK: st.global.u64 [%r[[R0]]], %rd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i64]* @array_i64, i32 0, i32 0
  store i64 %x, i64* %i
  ret void
}

define ptx_device void @t4_global_f32(float %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_float;
;CHECK: st.global.f32 [%r[[R0]]], %f{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x float]* @array_float, i32 0, i32 0
  store float %x, float* %i
  ret void
}

define ptx_device void @t4_global_f64(double %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_double;
;CHECK: st.global.f64 [%r[[R0]]], %fd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x double]* @array_double, i32 0, i32 0
  store double %x, double* %i
  ret void
}

define ptx_device void @t4_local_u16(i16 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_i16;
;CHECK: st.local.u16 [%r[[R0]]], %rh{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i16] addrspace(2)* @array_local_i16, i32 0, i32 0
  store i16 %x, i16 addrspace(2)* %i
  ret void
}

define ptx_device void @t4_local_u32(i32 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_i32;
;CHECK: st.local.u32 [%r[[R0]]], %r{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i32] addrspace(2)* @array_local_i32, i32 0, i32 0
  store i32 %x, i32 addrspace(2)* %i
  ret void
}

define ptx_device void @t4_local_u64(i64 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_i64;
;CHECK: st.local.u64 [%r[[R0]]], %rd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i64] addrspace(2)* @array_local_i64, i32 0, i32 0
  store i64 %x, i64 addrspace(2)* %i
  ret void
}

define ptx_device void @t4_local_f32(float %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_float;
;CHECK: st.local.f32 [%r[[R0]]], %f{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x float] addrspace(2)* @array_local_float, i32 0, i32 0
  store float %x, float addrspace(2)* %i
  ret void
}

define ptx_device void @t4_local_f64(double %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_double;
;CHECK: st.local.f64 [%r[[R0]]], %fd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x double] addrspace(2)* @array_local_double, i32 0, i32 0
  store double %x, double addrspace(2)* %i
  ret void
}

define ptx_device void @t4_shared_u16(i16 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_i16;
;CHECK: st.shared.u16 [%r[[R0]]], %rh{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i16] addrspace(4)* @array_shared_i16, i32 0, i32 0
  store i16 %x, i16 addrspace(4)* %i
  ret void
}

define ptx_device void @t4_shared_u32(i32 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_i32;
;CHECK: st.shared.u32 [%r[[R0]]], %r{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i32] addrspace(4)* @array_shared_i32, i32 0, i32 0
  store i32 %x, i32 addrspace(4)* %i
  ret void
}

define ptx_device void @t4_shared_u64(i64 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_i64;
;CHECK: st.shared.u64 [%r[[R0]]], %rd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i64] addrspace(4)* @array_shared_i64, i32 0, i32 0
  store i64 %x, i64 addrspace(4)* %i
  ret void
}

define ptx_device void @t4_shared_f32(float %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_float;
;CHECK: st.shared.f32 [%r[[R0]]], %f{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x float] addrspace(4)* @array_shared_float, i32 0, i32 0
  store float %x, float addrspace(4)* %i
  ret void
}

define ptx_device void @t4_shared_f64(double %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_double;
;CHECK: st.shared.f64 [%r[[R0]]], %fd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x double] addrspace(4)* @array_shared_double, i32 0, i32 0
  store double %x, double addrspace(4)* %i
  ret void
}

define ptx_device void @t5_u16(i16 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i16;
;CHECK: st.global.u16 [%r[[R0]]+2], %rh{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i16]* @array_i16, i32 0, i32 1
  store i16 %x, i16* %i
  ret void
}

define ptx_device void @t5_u32(i32 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i32;
;CHECK: st.global.u32 [%r[[R0]]+4], %r{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i32]* @array_i32, i32 0, i32 1
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t5_u64(i64 %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i64;
;CHECK: st.global.u64 [%r[[R0]]+8], %rd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x i64]* @array_i64, i32 0, i32 1
  store i64 %x, i64* %i
  ret void
}

define ptx_device void @t5_f32(float %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_float;
;CHECK: st.global.f32 [%r[[R0]]+4], %f{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x float]* @array_float, i32 0, i32 1
  store float %x, float* %i
  ret void
}

define ptx_device void @t5_f64(double %x) {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_double;
;CHECK: st.global.f64 [%r[[R0]]+8], %fd{{[0-9]+}};
;CHECK: ret;
  %i = getelementptr [10 x double]* @array_double, i32 0, i32 1
  store double %x, double* %i
  ret void
}
