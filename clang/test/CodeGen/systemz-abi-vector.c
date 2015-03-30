// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Vector types

typedef __attribute__((vector_size(1))) char v1i8;

typedef __attribute__((vector_size(2))) char v2i8;
typedef __attribute__((vector_size(2))) short v1i16;

typedef __attribute__((vector_size(4))) char v4i8;
typedef __attribute__((vector_size(4))) short v2i16;
typedef __attribute__((vector_size(4))) int v1i32;
typedef __attribute__((vector_size(4))) float v1f32;

typedef __attribute__((vector_size(8))) char v8i8;
typedef __attribute__((vector_size(8))) short v4i16;
typedef __attribute__((vector_size(8))) int v2i32;
typedef __attribute__((vector_size(8))) long long v1i64;
typedef __attribute__((vector_size(8))) float v2f32;
typedef __attribute__((vector_size(8))) double v1f64;

typedef __attribute__((vector_size(16))) char v16i8;
typedef __attribute__((vector_size(16))) short v8i16;
typedef __attribute__((vector_size(16))) int v4i32;
typedef __attribute__((vector_size(16))) long long v2i64;
typedef __attribute__((vector_size(16))) __int128 v1i128;
typedef __attribute__((vector_size(16))) float v4f32;
typedef __attribute__((vector_size(16))) double v2f64;
typedef __attribute__((vector_size(16))) long double v1f128;

typedef __attribute__((vector_size(32))) char v32i8;

v1i8 pass_v1i8(v1i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i8(<1 x i8>* noalias sret %{{.*}}, <1 x i8>*)

v2i8 pass_v2i8(v2i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i8(<2 x i8>* noalias sret %{{.*}}, <2 x i8>*)

v4i8 pass_v4i8(v4i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4i8(<4 x i8>* noalias sret %{{.*}}, <4 x i8>*)

v8i8 pass_v8i8(v8i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v8i8(<8 x i8>* noalias sret %{{.*}}, <8 x i8>*)

v16i8 pass_v16i8(v16i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v16i8(<16 x i8>* noalias sret %{{.*}}, <16 x i8>*)

v32i8 pass_v32i8(v32i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v32i8(<32 x i8>* noalias sret %{{.*}}, <32 x i8>*)

v1i16 pass_v1i16(v1i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i16(<1 x i16>* noalias sret %{{.*}}, <1 x i16>*)

v2i16 pass_v2i16(v2i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i16(<2 x i16>* noalias sret %{{.*}}, <2 x i16>*)

v4i16 pass_v4i16(v4i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4i16(<4 x i16>* noalias sret %{{.*}}, <4 x i16>*)

v8i16 pass_v8i16(v8i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v8i16(<8 x i16>* noalias sret %{{.*}}, <8 x i16>*)

v1i32 pass_v1i32(v1i32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i32(<1 x i32>* noalias sret %{{.*}}, <1 x i32>*)

v2i32 pass_v2i32(v2i32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i32(<2 x i32>* noalias sret %{{.*}}, <2 x i32>*)

v4i32 pass_v4i32(v4i32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4i32(<4 x i32>* noalias sret %{{.*}}, <4 x i32>*)

v1i64 pass_v1i64(v1i64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i64(<1 x i64>* noalias sret %{{.*}}, <1 x i64>*)

v2i64 pass_v2i64(v2i64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i64(<2 x i64>* noalias sret %{{.*}}, <2 x i64>*)

v1i128 pass_v1i128(v1i128 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i128(<1 x i128>* noalias sret %{{.*}}, <1 x i128>*)

v1f32 pass_v1f32(v1f32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1f32(<1 x float>* noalias sret %{{.*}}, <1 x float>*)

v2f32 pass_v2f32(v2f32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2f32(<2 x float>* noalias sret %{{.*}}, <2 x float>*)

v4f32 pass_v4f32(v4f32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4f32(<4 x float>* noalias sret %{{.*}}, <4 x float>*)

v1f64 pass_v1f64(v1f64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1f64(<1 x double>* noalias sret %{{.*}}, <1 x double>*)

v2f64 pass_v2f64(v2f64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2f64(<2 x double>* noalias sret %{{.*}}, <2 x double>*)

v1f128 pass_v1f128(v1f128 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1f128(<1 x fp128>* noalias sret %{{.*}}, <1 x fp128>*)


// Accessing variable argument lists

v1i8 va_v1i8(__builtin_va_list l) { return __builtin_va_arg(l, v1i8); }
// CHECK-LABEL: define void @va_v1i8(<1 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load <1 x i8>*

v2i8 va_v2i8(__builtin_va_list l) { return __builtin_va_arg(l, v2i8); }
// CHECK-LABEL: define void @va_v2i8(<2 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load <2 x i8>*

v4i8 va_v4i8(__builtin_va_list l) { return __builtin_va_arg(l, v4i8); }
// CHECK-LABEL: define void @va_v4i8(<4 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load <4 x i8>*

v8i8 va_v8i8(__builtin_va_list l) { return __builtin_va_arg(l, v8i8); }
// CHECK-LABEL: define void @va_v8i8(<8 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load <8 x i8>*

v16i8 va_v16i8(__builtin_va_list l) { return __builtin_va_arg(l, v16i8); }
// CHECK-LABEL: define void @va_v16i8(<16 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load <16 x i8>*

v32i8 va_v32i8(__builtin_va_list l) { return __builtin_va_arg(l, v32i8); }
// CHECK-LABEL: define void @va_v32i8(<32 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load <32 x i8>*

