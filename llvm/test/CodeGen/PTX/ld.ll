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


define ptx_device i16 @t1_u16(i16* %p) {
entry:
;CHECK: ld.global.u16 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
;CHECK: ret;
  %x = load i16* %p
  ret i16 %x
}

define ptx_device i32 @t1_u32(i32* %p) {
entry:
;CHECK: ld.global.u32 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
;CHECK: ret;
  %x = load i32* %p
  ret i32 %x
}

define ptx_device i64 @t1_u64(i64* %p) {
entry:
;CHECK: ld.global.u64 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
;CHECK: ret;
  %x = load i64* %p
  ret i64 %x
}

define ptx_device float @t1_f32(float* %p) {
entry:
;CHECK: ld.global.f32 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
;CHECK: ret;
  %x = load float* %p
  ret float %x
}

define ptx_device double @t1_f64(double* %p) {
entry:
;CHECK: ld.global.f64 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
;CHECK: ret;
  %x = load double* %p
  ret double %x
}

define ptx_device i16 @t2_u16(i16* %p) {
entry:
;CHECK: ld.global.u16 %ret{{[0-9]+}}, [%r{{[0-9]+}}+2];
;CHECK: ret;
  %i = getelementptr i16* %p, i32 1
  %x = load i16* %i
  ret i16 %x
}

define ptx_device i32 @t2_u32(i32* %p) {
entry:
;CHECK: ld.global.u32 %ret{{[0-9]+}}, [%r{{[0-9]+}}+4];
;CHECK: ret;
  %i = getelementptr i32* %p, i32 1
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i64 @t2_u64(i64* %p) {
entry:
;CHECK: ld.global.u64 %ret{{[0-9]+}}, [%r{{[0-9]+}}+8];
;CHECK: ret;
  %i = getelementptr i64* %p, i32 1
  %x = load i64* %i
  ret i64 %x
}

define ptx_device float @t2_f32(float* %p) {
entry:
;CHECK: ld.global.f32 %ret{{[0-9]+}}, [%r{{[0-9]+}}+4];
;CHECK: ret;
  %i = getelementptr float* %p, i32 1
  %x = load float* %i
  ret float %x
}

define ptx_device double @t2_f64(double* %p) {
entry:
;CHECK: ld.global.f64 %ret{{[0-9]+}}, [%r{{[0-9]+}}+8];
;CHECK: ret;
  %i = getelementptr double* %p, i32 1
  %x = load double* %i
  ret double %x
}

define ptx_device i16 @t3_u16(i16* %p, i32 %q) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 1;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: ld.global.u16 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
  %i = getelementptr i16* %p, i32 %q
  %x = load i16* %i
  ret i16 %x
}

define ptx_device i32 @t3_u32(i32* %p, i32 %q) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 2;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: ld.global.u32 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
  %i = getelementptr i32* %p, i32 %q
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i64 @t3_u64(i64* %p, i32 %q) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 3;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: ld.global.u64 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
  %i = getelementptr i64* %p, i32 %q
  %x = load i64* %i
  ret i64 %x
}

define ptx_device float @t3_f32(float* %p, i32 %q) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 2;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: ld.global.f32 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
  %i = getelementptr float* %p, i32 %q
  %x = load float* %i
  ret float %x
}

define ptx_device double @t3_f64(double* %p, i32 %q) {
entry:
;CHECK: shl.b32 %r[[R0:[0-9]+]], %r{{[0-9]+}}, 3;
;CHECK: add.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r[[R0]];
;CHECK: ld.global.f64 %ret{{[0-9]+}}, [%r{{[0-9]+}}];
  %i = getelementptr double* %p, i32 %q
  %x = load double* %i
  ret double %x
}

define ptx_device i16 @t4_global_u16() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i16;
;CHECK: ld.global.u16 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i16]* @array_i16, i32 0, i32 0
  %x = load i16* %i
  ret i16 %x
}

define ptx_device i32 @t4_global_u32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i32;
;CHECK: ld.global.u32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i32]* @array_i32, i32 0, i32 0
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i64 @t4_global_u64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i64;
;CHECK: ld.global.u64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i64]* @array_i64, i32 0, i32 0
  %x = load i64* %i
  ret i64 %x
}

define ptx_device float @t4_global_f32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_float;
;CHECK: ld.global.f32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x float]* @array_float, i32 0, i32 0
  %x = load float* %i
  ret float %x
}

define ptx_device double @t4_global_f64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_double;
;CHECK: ld.global.f64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x double]* @array_double, i32 0, i32 0
  %x = load double* %i
  ret double %x
}

define ptx_device i16 @t4_const_u16() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_constant_i16;
;CHECK: ld.const.u16 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i16] addrspace(1)* @array_constant_i16, i32 0, i32 0
  %x = load i16 addrspace(1)* %i
  ret i16 %x
}

define ptx_device i32 @t4_const_u32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_constant_i32;
;CHECK: ld.const.u32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i32] addrspace(1)* @array_constant_i32, i32 0, i32 0
  %x = load i32 addrspace(1)* %i
  ret i32 %x
}

define ptx_device i64 @t4_const_u64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_constant_i64;
;CHECK: ld.const.u64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i64] addrspace(1)* @array_constant_i64, i32 0, i32 0
  %x = load i64 addrspace(1)* %i
  ret i64 %x
}

define ptx_device float @t4_const_f32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_constant_float;
;CHECK: ld.const.f32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x float] addrspace(1)* @array_constant_float, i32 0, i32 0
  %x = load float addrspace(1)* %i
  ret float %x
}

define ptx_device double @t4_const_f64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_constant_double;
;CHECK: ld.const.f64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x double] addrspace(1)* @array_constant_double, i32 0, i32 0
  %x = load double addrspace(1)* %i
  ret double %x
}

define ptx_device i16 @t4_local_u16() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_i16;
;CHECK: ld.local.u16 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i16] addrspace(2)* @array_local_i16, i32 0, i32 0
  %x = load i16 addrspace(2)* %i
  ret i16 %x
}

define ptx_device i32 @t4_local_u32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_i32;
;CHECK: ld.local.u32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i32] addrspace(2)* @array_local_i32, i32 0, i32 0
  %x = load i32 addrspace(2)* %i
  ret i32 %x
}

define ptx_device i64 @t4_local_u64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_i64;
;CHECK: ld.local.u64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i64] addrspace(2)* @array_local_i64, i32 0, i32 0
  %x = load i64 addrspace(2)* %i
  ret i64 %x
}

define ptx_device float @t4_local_f32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_float;
;CHECK: ld.local.f32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x float] addrspace(2)* @array_local_float, i32 0, i32 0
  %x = load float addrspace(2)* %i
  ret float %x
}

define ptx_device double @t4_local_f64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_local_double;
;CHECK: ld.local.f64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x double] addrspace(2)* @array_local_double, i32 0, i32 0
  %x = load double addrspace(2)* %i
  ret double %x
}

define ptx_device i16 @t4_shared_u16() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_i16;
;CHECK: ld.shared.u16 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i16] addrspace(4)* @array_shared_i16, i32 0, i32 0
  %x = load i16 addrspace(4)* %i
  ret i16 %x
}

define ptx_device i32 @t4_shared_u32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_i32;
;CHECK: ld.shared.u32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i32] addrspace(4)* @array_shared_i32, i32 0, i32 0
  %x = load i32 addrspace(4)* %i
  ret i32 %x
}

define ptx_device i64 @t4_shared_u64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_i64;
;CHECK: ld.shared.u64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x i64] addrspace(4)* @array_shared_i64, i32 0, i32 0
  %x = load i64 addrspace(4)* %i
  ret i64 %x
}

define ptx_device float @t4_shared_f32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_float;
;CHECK: ld.shared.f32 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x float] addrspace(4)* @array_shared_float, i32 0, i32 0
  %x = load float addrspace(4)* %i
  ret float %x
}

define ptx_device double @t4_shared_f64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_shared_double;
;CHECK: ld.shared.f64 %ret{{[0-9]+}}, [%r[[R0]]];
;CHECK: ret;
  %i = getelementptr [10 x double] addrspace(4)* @array_shared_double, i32 0, i32 0
  %x = load double addrspace(4)* %i
  ret double %x
}

define ptx_device i16 @t5_u16() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i16;
;CHECK: ld.global.u16 %ret{{[0-9]+}}, [%r[[R0]]+2];
;CHECK: ret;
  %i = getelementptr [10 x i16]* @array_i16, i32 0, i32 1
  %x = load i16* %i
  ret i16 %x
}

define ptx_device i32 @t5_u32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i32;
;CHECK: ld.global.u32 %ret{{[0-9]+}}, [%r[[R0]]+4];
;CHECK: ret;
  %i = getelementptr [10 x i32]* @array_i32, i32 0, i32 1
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i64 @t5_u64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_i64;
;CHECK: ld.global.u64 %ret{{[0-9]+}}, [%r[[R0]]+8];
;CHECK: ret;
  %i = getelementptr [10 x i64]* @array_i64, i32 0, i32 1
  %x = load i64* %i
  ret i64 %x
}

define ptx_device float @t5_f32() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_float;
;CHECK: ld.global.f32 %ret{{[0-9]+}}, [%r[[R0]]+4];
;CHECK: ret;
  %i = getelementptr [10 x float]* @array_float, i32 0, i32 1
  %x = load float* %i
  ret float %x
}

define ptx_device double @t5_f64() {
entry:
;CHECK: mov.u32 %r[[R0:[0-9]+]], array_double;
;CHECK: ld.global.f64 %ret{{[0-9]+}}, [%r[[R0]]+8];
;CHECK: ret;
  %i = getelementptr [10 x double]* @array_double, i32 0, i32 1
  %x = load double* %i
  ret double %x
}
