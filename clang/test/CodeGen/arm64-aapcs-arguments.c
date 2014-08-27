// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-abi aapcs -ffreestanding -fallow-half-arguments-and-returns -emit-llvm -w -o - %s | FileCheck %s

// AAPCS clause C.8 says: If the argument has an alignment of 16 then the NGRN
// is rounded up to the next even number.

// CHECK: void @test1(i32 %x0, i128 %x2_x3, i128 %x4_x5, i128 %x6_x7, i128 %sp.coerce)
typedef union { __int128 a; } Small;
void test1(int x0, __int128 x2_x3, __int128 x4_x5, __int128 x6_x7, Small sp) {
}


// CHECK: void @test2(i32 %x0, i128 %x2_x3.coerce, i32 %x4, i128 %x6_x7.coerce, i32 %sp, i128 %sp16.coerce)
void test2(int x0, Small x2_x3, int x4, Small x6_x7, int sp, Small sp16) {
}

// We coerce HFAs into a contiguous [N x double] type if they're going on the
// stack in order to avoid holes. Make sure we get all of them, and not just the
// first:

// CHECK: void @test3(float %s0_s3.0, float %s0_s3.1, float %s0_s3.2, float %s0_s3.3, float %s4, [3 x float], [2 x double] %sp.coerce, [2 x double] %sp16.coerce)
typedef struct { float arr[4]; } HFA;
void test3(HFA s0_s3, float s4, HFA sp, HFA sp16) {
}


// However, we shouldn't perform the [N x double] coercion on types which have
// sufficient alignment to avoid holes on their own. We could coerce to [N x
// fp128] or something, but leaving them as-is retains more information for
// users to debug.

//  CHECK: void @test4(<16 x i8> %v0_v2.0, <16 x i8> %v0_v2.1, <16 x i8> %v0_v2.2, <16 x i8> %v3_v5.0, <16 x i8> %v3_v5.1, <16 x i8> %v3_v5.2, [2 x float], <16 x i8> %sp.0, <16 x i8> %sp.1, <16 x i8> %sp.2, double %sp48, <16 x i8> %sp64.0, <16 x i8> %sp64.1, <16 x i8> %sp64.2)
typedef __attribute__((neon_vector_type(16))) signed char int8x16_t;
typedef struct { int8x16_t arr[3]; } BigHFA;
void test4(BigHFA v0_v2, BigHFA v3_v5, BigHFA sp, double sp48, BigHFA sp64) {
}

// It's the job of the argument *consumer* to perform the required sign & zero
// extensions under AAPCS. There shouldn't be

// CHECK: define i8 @test5(i8 %a, i16 %b)
unsigned char test5(unsigned char a, signed short b) {
}

// __fp16 can be used as a function argument or return type (ACLE 2.0)
// CHECK: define half @test_half(half %{{.*}})
__fp16 test_half(__fp16 A) { }

// __fp16 is a base type for homogeneous floating-point aggregates for AArch64 (but not 32-bit ARM).
// CHECK: define %struct.HFA_half @test_half_hfa(half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}})
struct HFA_half { __fp16 a[4]; };
struct HFA_half test_half_hfa(struct HFA_half A) { }
