// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - %s | FileCheck %s
// REQUIRES: arm64-registered-target
/// Test vqmov[u]n_high_<su>{16,32,64) ARM64 intrinsics

#include <arm_neon.h>

// vqmovn_high_s16 -> UQXTN2 Vd.16b,Vn.8h
int8x16_t test_vqmovn_high_s16(int8x8_t Vdlow, int16x8_t Vn)
{
    return vqmovn_high_s16(Vdlow, Vn);
  // CHECK: test_vqmovn_high_s16:
  // CHECK: sqxtn2.16b {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovun_high_s16 -> UQXTN2 Vd.16b,Vn.8h
uint8x16_t test_vqmovun_high_s16(uint8x8_t Vdlow, uint16x8_t Vn)
{
    return vqmovun_high_s16(Vdlow, Vn);
  // CHECK: test_vqmovun_high_s16:
  // CHECK: sqxtun2.16b {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovn_high_s32 -> SQXTN2 Vd.8h,Vn.4s
int16x8_t test_vqmovn_high_s32(int16x4_t Vdlow, int32x4_t Vn)
{
    return vqmovn_high_s32(Vdlow, Vn);
  // CHECK: test_vqmovn_high_s32:
  // CHECK: sqxtn2.8h {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovn_high_u32 -> UQXTN2 Vd.8h,Vn.4s
uint16x8_t test_vqmovn_high_u32(uint16x4_t Vdlow, uint32x4_t Vn)
{
    return vqmovn_high_u32(Vdlow, Vn);
  // CHECK: test_vqmovn_high_u32:
  // CHECK: uqxtn2.8h {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovn_high_s64 -> SQXTN2 Vd.4s,Vn.2d
int32x4_t test_vqmovn_high_s64(int32x2_t Vdlow, int64x2_t Vn)
{
    return vqmovn_high_s64(Vdlow, Vn);
  // CHECK: test_vqmovn_high_s64:
  // CHECK: sqxtn2.4s {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovn_high_u64 -> UQXTN2 Vd.4s,Vn.2d
uint32x4_t test_vqmovn_high_u64(uint32x2_t Vdlow, uint64x2_t Vn)
{
    return vqmovn_high_u64(Vdlow, Vn);
  // CHECK: test_vqmovn_high_u64:
  // CHECK: uqxtn2.4s {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovn_high_u16 -> UQXTN2 Vd.16b,Vn.8h
uint8x16_t test_vqmovn_high_u16(uint8x8_t Vdlow, uint16x8_t Vn)
{
    return vqmovn_high_u16(Vdlow, Vn);
  // CHECK: test_vqmovn_high_u16:
  // CHECK: uqxtn2.16b {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovun_high_s32 -> SQXTUN2 Vd.8h,Vn.4s
uint16x8_t test_vqmovun_high_s32(uint16x4_t Vdlow, uint32x4_t Vn)
{
    return vqmovun_high_s32(Vdlow, Vn);
  // CHECK: test_vqmovun_high_s32:
  // CHECK: sqxtun2.8h {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}

// vqmovun_high_s64 -> SQXTUN2  Vd.4s,Vn.2d
uint32x4_t test_vqmovun_high_s64(uint32x2_t Vdlow, uint64x2_t Vn)
{
    return vqmovun_high_s64(Vdlow, Vn);
  // CHECK: test_vqmovun_high_s64:
  // CHECK: sqxtun2.4s {{v[0-9][0-9]*}}, {{v[0-9][0-9]*}}
}
