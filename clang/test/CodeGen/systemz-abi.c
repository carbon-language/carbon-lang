// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Scalar types

char pass_char(char arg) { return arg; }
// CHECK-LABEL: define signext i8 @pass_char(i8 signext %{{.*}})

short pass_short(short arg) { return arg; }
// CHECK-LABEL: define signext i16 @pass_short(i16 signext %{{.*}})

int pass_int(int arg) { return arg; }
// CHECK-LABEL: define signext i32 @pass_int(i32 signext %{{.*}})

long pass_long(long arg) { return arg; }
// CHECK-LABEL: define i64 @pass_long(i64 %{{.*}})

long long pass_longlong(long long arg) { return arg; }
// CHECK-LABEL: define i64 @pass_longlong(i64 %{{.*}})

__int128 pass_int128(__int128 arg) { return arg; }
// CHECK-LABEL: define void @pass_int128(i128* noalias sret %{{.*}}, i128*)

float pass_float(float arg) { return arg; }
// CHECK-LABEL: define float @pass_float(float %{{.*}})

double pass_double(double arg) { return arg; }
// CHECK-LABEL: define double @pass_double(double %{{.*}})

long double pass_longdouble(long double arg) { return arg; }
// CHECK-LABEL: define void @pass_longdouble(fp128* noalias sret %{{.*}}, fp128*)


// Complex types

_Complex char pass_complex_char(_Complex char arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_char({ i8, i8 }* noalias sret %{{.*}}, { i8, i8 }* %{{.*}}arg)

_Complex short pass_complex_short(_Complex short arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_short({ i16, i16 }* noalias sret %{{.*}}, { i16, i16 }* %{{.*}}arg)

_Complex int pass_complex_int(_Complex int arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_int({ i32, i32 }* noalias sret %{{.*}}, { i32, i32 }* %{{.*}}arg)

_Complex long pass_complex_long(_Complex long arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_long({ i64, i64 }* noalias sret %{{.*}}, { i64, i64 }* %{{.*}}arg)

_Complex long long pass_complex_longlong(_Complex long long arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_longlong({ i64, i64 }* noalias sret %{{.*}}, { i64, i64 }* %{{.*}}arg)

_Complex float pass_complex_float(_Complex float arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_float({ float, float }* noalias sret %{{.*}}, { float, float }* %{{.*}}arg)

_Complex double pass_complex_double(_Complex double arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_double({ double, double }* noalias sret %{{.*}}, { double, double }* %{{.*}}arg)

_Complex long double pass_complex_longdouble(_Complex long double arg) { return arg; }
// CHECK-LABEL: define void @pass_complex_longdouble({ fp128, fp128 }* noalias sret %{{.*}}, { fp128, fp128 }* %{{.*}}arg)


// Aggregate types

struct agg_1byte { char a[1]; };
struct agg_1byte pass_agg_1byte(struct agg_1byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_1byte(%struct.agg_1byte* noalias sret %{{.*}}, i8 %{{.*}})

struct agg_2byte { char a[2]; };
struct agg_2byte pass_agg_2byte(struct agg_2byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_2byte(%struct.agg_2byte* noalias sret %{{.*}}, i16 %{{.*}})

struct agg_3byte { char a[3]; };
struct agg_3byte pass_agg_3byte(struct agg_3byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_3byte(%struct.agg_3byte* noalias sret %{{.*}}, %struct.agg_3byte* %{{.*}})

struct agg_4byte { char a[4]; };
struct agg_4byte pass_agg_4byte(struct agg_4byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_4byte(%struct.agg_4byte* noalias sret %{{.*}}, i32 %{{.*}})

struct agg_5byte { char a[5]; };
struct agg_5byte pass_agg_5byte(struct agg_5byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_5byte(%struct.agg_5byte* noalias sret %{{.*}}, %struct.agg_5byte* %{{.*}})

struct agg_6byte { char a[6]; };
struct agg_6byte pass_agg_6byte(struct agg_6byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_6byte(%struct.agg_6byte* noalias sret %{{.*}}, %struct.agg_6byte* %{{.*}})

struct agg_7byte { char a[7]; };
struct agg_7byte pass_agg_7byte(struct agg_7byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_7byte(%struct.agg_7byte* noalias sret %{{.*}}, %struct.agg_7byte* %{{.*}})

struct agg_8byte { char a[8]; };
struct agg_8byte pass_agg_8byte(struct agg_8byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_8byte(%struct.agg_8byte* noalias sret %{{.*}}, i64 %{{.*}})

struct agg_16byte { char a[16]; };
struct agg_16byte pass_agg_16byte(struct agg_16byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_16byte(%struct.agg_16byte* noalias sret %{{.*}}, %struct.agg_16byte* %{{.*}})


// Float-like aggregate types

struct agg_float { float a; };
struct agg_float pass_agg_float(struct agg_float arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_float(%struct.agg_float* noalias sret %{{.*}}, float %{{.*}})

struct agg_double { double a; };
struct agg_double pass_agg_double(struct agg_double arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_double(%struct.agg_double* noalias sret %{{.*}}, double %{{.*}})

struct agg_longdouble { long double a; };
struct agg_longdouble pass_agg_longdouble(struct agg_longdouble arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_longdouble(%struct.agg_longdouble* noalias sret %{{.*}}, %struct.agg_longdouble* %{{.*}})

struct agg_float_a8 { float a __attribute__((aligned (8))); };
struct agg_float_a8 pass_agg_float_a8(struct agg_float_a8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_float_a8(%struct.agg_float_a8* noalias sret %{{.*}}, double %{{.*}})

struct agg_float_a16 { float a __attribute__((aligned (16))); };
struct agg_float_a16 pass_agg_float_a16(struct agg_float_a16 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_float_a16(%struct.agg_float_a16* noalias sret %{{.*}}, %struct.agg_float_a16* %{{.*}})


// Verify that the following are *not* float-like aggregate types

struct agg_nofloat1 { float a; float b; };
struct agg_nofloat1 pass_agg_nofloat1(struct agg_nofloat1 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_nofloat1(%struct.agg_nofloat1* noalias sret %{{.*}}, i64 %{{.*}})

struct agg_nofloat2 { float a; int b; };
struct agg_nofloat2 pass_agg_nofloat2(struct agg_nofloat2 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_nofloat2(%struct.agg_nofloat2* noalias sret %{{.*}}, i64 %{{.*}})

struct agg_nofloat3 { float a; int : 0; };
struct agg_nofloat3 pass_agg_nofloat3(struct agg_nofloat3 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_nofloat3(%struct.agg_nofloat3* noalias sret %{{.*}}, i32 %{{.*}})


// Accessing variable argument lists

int va_int(__builtin_va_list l) { return __builtin_va_arg(l, int); }
// CHECK-LABEL: define signext i32 @va_int(%struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 20
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 4
// CHECK-NOT: %indirect_arg

long va_long(__builtin_va_list l) { return __builtin_va_arg(l, long); }
// CHECK-LABEL: define i64 @va_long(%struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

long long va_longlong(__builtin_va_list l) { return __builtin_va_arg(l, long long); }
// CHECK-LABEL: define i64 @va_longlong(%struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

double va_double(__builtin_va_list l) { return __builtin_va_arg(l, double); }
// CHECK-LABEL: define double @va_double(%struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 128
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

long double va_longdouble(__builtin_va_list l) { return __builtin_va_arg(l, long double); }
// CHECK-LABEL: define void @va_longdouble(fp128* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load fp128*

_Complex char va_complex_char(__builtin_va_list l) { return __builtin_va_arg(l, _Complex char); }
// CHECK-LABEL: define void @va_complex_char({ i8, i8 }* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load { i8, i8 }*

struct agg_1byte va_agg_1byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_1byte); }
// CHECK-LABEL: define void @va_agg_1byte(%struct.agg_1byte* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 23
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 7
// CHECK-NOT: %indirect_arg

struct agg_2byte va_agg_2byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_2byte); }
// CHECK-LABEL: define void @va_agg_2byte(%struct.agg_2byte* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 22
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 6
// CHECK-NOT: %indirect_arg

struct agg_3byte va_agg_3byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_3byte); }
// CHECK-LABEL: define void @va_agg_3byte(%struct.agg_3byte* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load %struct.agg_3byte*

struct agg_4byte va_agg_4byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_4byte); }
// CHECK-LABEL: define void @va_agg_4byte(%struct.agg_4byte* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 20
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 4
// CHECK-NOT: %indirect_arg

struct agg_8byte va_agg_8byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_8byte); }
// CHECK-LABEL: define void @va_agg_8byte(%struct.agg_8byte* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

struct agg_float va_agg_float(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_float); }
// CHECK-LABEL: define void @va_agg_float(%struct.agg_float* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 128
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 4
// CHECK-NOT: %indirect_arg

struct agg_double va_agg_double(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_double); }
// CHECK-LABEL: define void @va_agg_double(%struct.agg_double* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 128
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

struct agg_longdouble va_agg_longdouble(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_longdouble); }
// CHECK-LABEL: define void @va_agg_longdouble(%struct.agg_longdouble* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load %struct.agg_longdouble*

struct agg_float_a8 va_agg_float_a8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_float_a8); }
// CHECK-LABEL: define void @va_agg_float_a8(%struct.agg_float_a8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 128
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

struct agg_float_a16 va_agg_float_a16(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_float_a16); }
// CHECK-LABEL: define void @va_agg_float_a16(%struct.agg_float_a16* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK: %indirect_arg = load %struct.agg_float_a16*

struct agg_nofloat1 va_agg_nofloat1(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_nofloat1); }
// CHECK-LABEL: define void @va_agg_nofloat1(%struct.agg_nofloat1* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

struct agg_nofloat2 va_agg_nofloat2(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_nofloat2); }
// CHECK-LABEL: define void @va_agg_nofloat2(%struct.agg_nofloat2* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 16
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 0
// CHECK-NOT: %indirect_arg

struct agg_nofloat3 va_agg_nofloat3(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_nofloat3); }
// CHECK-LABEL: define void @va_agg_nofloat3(%struct.agg_nofloat3* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: %reg_offset = add i64 %scaled_reg_count, 20
// CHECK: %raw_mem_addr = getelementptr i8, i8* %overflow_arg_area, i64 4
// CHECK-NOT: %indirect_arg

