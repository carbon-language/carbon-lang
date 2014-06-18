// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-feature +neon  \
// RUN:   -S -O1 -o - -ffreestanding %s | FileCheck %s

// We're explicitly using arm_neon.h here: some types probably don't match
// the ACLE definitions, but we want to check current codegen.
#include <arm_neon.h>

float test_vrsqrtss_f32(float a, float b) {
// CHECK: test_vrsqrtss_f32
  return vrsqrtss_f32(a, b);
// CHECK: frsqrts {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

double test_vrsqrtsd_f64(double a, double b) {
// CHECK: test_vrsqrtsd_f64
  return vrsqrtsd_f64(a, b);
// CHECK: frsqrts {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int64x1_t test_vrshl_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vrshl_s64
  return vrshl_s64(a, b);
// CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vrshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK: test_vrshl_u64
  return vrshl_u64(a, b);
// CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vrshld_s64
int64_t test_vrshld_s64(int64_t a, int64_t b) {
  return vrshld_s64(a, b);
// CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vrshld_u64
uint64_t test_vrshld_u64(uint64_t a, uint64_t b) {
  return vrshld_u64(a, b);
// CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqrshlb_s8
int8_t test_vqrshlb_s8(int8_t a, int8_t b) {
  return vqrshlb_s8(a, b);
// CHECK: sqrshl.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqrshlh_s16
int16_t test_vqrshlh_s16(int16_t a, int16_t b) {
  return vqrshlh_s16(a, b);
// CHECK: sqrshl.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqrshls_s32
int32_t test_vqrshls_s32(int32_t a, int32_t b) {
  return vqrshls_s32(a, b);
// CHECK: sqrshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqrshld_s64
int64_t test_vqrshld_s64(int64_t a, int64_t b) {
  return vqrshld_s64(a, b);
// CHECK: sqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqrshlb_u8
uint8_t test_vqrshlb_u8(uint8_t a, uint8_t b) {
  return vqrshlb_u8(a, b);
// CHECK: uqrshl.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqrshlh_u16
uint16_t test_vqrshlh_u16(uint16_t a, uint16_t b) {
  return vqrshlh_u16(a, b);
// CHECK: uqrshl.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqrshls_u32
uint32_t test_vqrshls_u32(uint32_t a, uint32_t b) {
  return vqrshls_u32(a, b);
// CHECK: uqrshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqrshld_u64
uint64_t test_vqrshld_u64(uint64_t a, uint64_t b) {
  return vqrshld_u64(a, b);
// CHECK: uqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqshlb_s8
int8_t test_vqshlb_s8(int8_t a, int8_t b) {
  return vqshlb_s8(a, b);
// CHECK: sqshl.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqshlh_s16
int16_t test_vqshlh_s16(int16_t a, int16_t b) {
  return vqshlh_s16(a, b);
// CHECK: sqshl.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqshls_s32
int32_t test_vqshls_s32(int32_t a, int32_t b) {
  return vqshls_s32(a, b);
// CHECK: sqshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqshld_s64
int64_t test_vqshld_s64(int64_t a, int64_t b) {
  return vqshld_s64(a, b);
// CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqshld_s64_i
int64_t test_vqshld_s64_i(int64_t a) {
  return vqshld_s64(a, 36);
// CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, #36
}

// CHECK: test_vqshlb_u8
uint8_t test_vqshlb_u8(uint8_t a, uint8_t b) {
  return vqshlb_u8(a, b);
// CHECK: uqshl.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqshlh_u16
uint16_t test_vqshlh_u16(uint16_t a, uint16_t b) {
  return vqshlh_u16(a, b);
// CHECK: uqshl.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqshls_u32
uint32_t test_vqshls_u32(uint32_t a, uint32_t b) {
  return vqshls_u32(a, b);
// CHECK: uqshl {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqshld_u64
uint64_t test_vqshld_u64(uint64_t a, uint64_t b) {
  return vqshld_u64(a, b);
// CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqshld_u64_i
uint64_t test_vqshld_u64_i(uint64_t a) {
  return vqshld_u64(a, 36);
// CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, #36
}

// CHECK: test_vshld_u64
uint64_t test_vshld_u64(uint64_t a, uint64_t b) {
  return vshld_u64(a, b);
// CHECK: ushl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vshld_s64
int64_t test_vshld_s64(int64_t a, int64_t b) {
  return vshld_s64(a, b);
// CHECK: sshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqdmullh_s16
int32_t test_vqdmullh_s16(int16_t a, int16_t b) {
  return vqdmullh_s16(a, b);
// CHECK: sqdmull.4s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqdmulls_s32
int64_t test_vqdmulls_s32(int32_t a, int32_t b) {
  return vqdmulls_s32(a, b);
// CHECK: sqdmull {{d[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqaddb_s8
int8_t test_vqaddb_s8(int8_t a, int8_t b) {
  return vqaddb_s8(a, b);
// CHECK: sqadd.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqaddh_s16
int16_t test_vqaddh_s16(int16_t a, int16_t b) {
  return vqaddh_s16(a, b);
// CHECK: sqadd.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqadds_s32
int32_t test_vqadds_s32(int32_t a, int32_t b) {
  return vqadds_s32(a, b);
// CHECK: sqadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqaddd_s64
int64_t test_vqaddd_s64(int64_t a, int64_t b) {
  return vqaddd_s64(a, b);
// CHECK: sqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqaddb_u8
uint8_t test_vqaddb_u8(uint8_t a, uint8_t b) {
  return vqaddb_u8(a, b);
// CHECK: uqadd.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqaddh_u16
uint16_t test_vqaddh_u16(uint16_t a, uint16_t b) {
  return vqaddh_u16(a, b);
// CHECK: uqadd.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqadds_u32
uint32_t test_vqadds_u32(uint32_t a, uint32_t b) {
  return vqadds_u32(a, b);
// CHECK: uqadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqaddd_u64
uint64_t test_vqaddd_u64(uint64_t a, uint64_t b) {
  return vqaddd_u64(a, b);
// CHECK: uqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqsubb_s8
int8_t test_vqsubb_s8(int8_t a, int8_t b) {
  return vqsubb_s8(a, b);
// CHECK: sqsub.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqsubh_s16
int16_t test_vqsubh_s16(int16_t a, int16_t b) {
  return vqsubh_s16(a, b);
// CHECK: sqsub.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqsubs_s32
int32_t test_vqsubs_s32(int32_t a, int32_t b) {
  return vqsubs_s32(a, b);
// CHECK: sqsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqsubd_s64
int64_t test_vqsubd_s64(int64_t a, int64_t b) {
  return vqsubd_s64(a, b);
// CHECK: sqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqsubb_u8
uint8_t test_vqsubb_u8(uint8_t a, uint8_t b) {
  return vqsubb_u8(a, b);
// CHECK: uqsub.8b {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqsubh_u16
uint16_t test_vqsubh_u16(uint16_t a, uint16_t b) {
  return vqsubh_u16(a, b);
// CHECK: uqsub.4h {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqsubs_u32
uint32_t test_vqsubs_u32(uint32_t a, uint32_t b) {
  return vqsubs_u32(a, b);
// CHECK: uqsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqsubd_u64
uint64_t test_vqsubd_u64(uint64_t a, uint64_t b) {
  return vqsubd_u64(a, b);
// CHECK: uqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqmovnh_s16
int8_t test_vqmovnh_s16(int16_t a) {
  return vqmovnh_s16(a);
// CHECK: sqxtn.8b {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqmovnh_u16
uint8_t test_vqmovnh_u16(uint16_t a) {
  return vqmovnh_u16(a);
// CHECK: uqxtn.8b {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqmovns_s32
int16_t test_vqmovns_s32(int32_t a) {
  return vqmovns_s32(a);
// CHECK: sqxtn.4h {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqmovns_u32
uint16_t test_vqmovns_u32(uint32_t a) {
  return vqmovns_u32(a);
// CHECK: uqxtn.4h {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqmovnd_s64
int32_t test_vqmovnd_s64(int64_t a) {
  return vqmovnd_s64(a);
// CHECK: sqxtn {{s[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqmovnd_u64
uint32_t test_vqmovnd_u64(uint64_t a) {
  return vqmovnd_u64(a);
// CHECK: uqxtn {{s[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqmovunh_s16
int8_t test_vqmovunh_s16(int16_t a) {
  return vqmovunh_s16(a);
// CHECK: sqxtun.8b {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqmovuns_s32
int16_t test_vqmovuns_s32(int32_t a) {
  return vqmovuns_s32(a);
// CHECK: sqxtun.4h {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqmovund_s64
int32_t test_vqmovund_s64(int64_t a) {
  return vqmovund_s64(a);
// CHECK: sqxtun {{s[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqabsb_s8
int8_t test_vqabsb_s8(int8_t a) {
  return vqabsb_s8(a);
// CHECK: sqabs.8b {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqabsh_s16
int16_t test_vqabsh_s16(int16_t a) {
  return vqabsh_s16(a);
// CHECK: sqabs.4h {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqabss_s32
int32_t test_vqabss_s32(int32_t a) {
  return vqabss_s32(a);
// CHECK: sqabs {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqabsd_s64
int64_t test_vqabsd_s64(int64_t a) {
  return vqabsd_s64(a);
// CHECK: sqabs {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vqnegb_s8
int8_t test_vqnegb_s8(int8_t a) {
  return vqnegb_s8(a);
// CHECK: sqneg.8b {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqnegh_s16
int16_t test_vqnegh_s16(int16_t a) {
  return vqnegh_s16(a);
// CHECK: sqneg.4h {{v[0-9]+}}, {{v[0-9]+}}
}

// CHECK: test_vqnegs_s32
int32_t test_vqnegs_s32(int32_t a) {
  return vqnegs_s32(a);
// CHECK: sqneg {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vqnegd_s64
int64_t test_vqnegd_s64(int64_t a) {
  return vqnegd_s64(a);
// CHECK: sqneg {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvts_n_f32_s32
float32_t test_vcvts_n_f32_s32(int32_t a) {
  return vcvts_n_f32_s32(a, 3);
// CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}, #3
}

// CHECK: test_vcvts_n_f32_u32
float32_t test_vcvts_n_f32_u32(uint32_t a) {
  return vcvts_n_f32_u32(a, 3);
// CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}, #3
}

// CHECK: test_vcvtd_n_f64_s64
float64_t test_vcvtd_n_f64_s64(int64_t a) {
  return vcvtd_n_f64_s64(a, 3);
// CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}, #3
}

// CHECK: test_vcvtd_n_f64_u64
float64_t test_vcvtd_n_f64_u64(uint64_t a) {
  return vcvtd_n_f64_u64(a, 3);
// CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}, #3
}

// CHECK: test_vcvts_n_s32_f32
int32_t test_vcvts_n_s32_f32(float32_t a) {
  return vcvts_n_s32_f32(a, 3);
// CHECK: fcvtzs {{s[0-9]+}}, {{s[0-9]+}}, #3
}

// CHECK: test_vcvts_n_u32_f32
uint32_t test_vcvts_n_u32_f32(float32_t a) {
  return vcvts_n_u32_f32(a, 3);
// CHECK: fcvtzu {{s[0-9]+}}, {{s[0-9]+}}, #3
}

// CHECK: test_vcvtd_n_s64_f64
int64_t test_vcvtd_n_s64_f64(float64_t a) {
  return vcvtd_n_s64_f64(a, 3);
// CHECK: fcvtzs {{d[0-9]+}}, {{d[0-9]+}}, #3
}

// CHECK: test_vcvtd_n_u64_f64
uint64_t test_vcvtd_n_u64_f64(float64_t a) {
  return vcvtd_n_u64_f64(a, 3);
// CHECK: fcvtzu {{d[0-9]+}}, {{d[0-9]+}}, #3
}

// CHECK: test_vcvtas_s32_f32
int32_t test_vcvtas_s32_f32(float32_t a) {
  return vcvtas_s32_f32(a);
// CHECK: fcvtas {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtas_u32_f32
uint32_t test_vcvtas_u32_f32(float32_t a) {
  return vcvtas_u32_f32(a);
// CHECK: fcvtau {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtad_s64_f64
int64_t test_vcvtad_s64_f64(float64_t a) {
  return vcvtad_s64_f64(a);
// CHECK: fcvtas {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtad_u64_f64
uint64_t test_vcvtad_u64_f64(float64_t a) {
  return vcvtad_u64_f64(a);
// CHECK: fcvtau {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtms_s32_f32
int32_t test_vcvtms_s32_f32(float32_t a) {
  return vcvtms_s32_f32(a);
// CHECK: fcvtms {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtms_u32_f32
uint32_t test_vcvtms_u32_f32(float32_t a) {
  return vcvtms_u32_f32(a);
// CHECK: fcvtmu {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtmd_s64_f64
int64_t test_vcvtmd_s64_f64(float64_t a) {
  return vcvtmd_s64_f64(a);
// CHECK: fcvtms {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtmd_u64_f64
uint64_t test_vcvtmd_u64_f64(float64_t a) {
  return vcvtmd_u64_f64(a);
// CHECK: fcvtmu {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtns_s32_f32
int32_t test_vcvtns_s32_f32(float32_t a) {
  return vcvtns_s32_f32(a);
// CHECK: fcvtns {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtns_u32_f32
uint32_t test_vcvtns_u32_f32(float32_t a) {
  return vcvtns_u32_f32(a);
// CHECK: fcvtnu {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtnd_s64_f64
int64_t test_vcvtnd_s64_f64(float64_t a) {
  return vcvtnd_s64_f64(a);
// CHECK: fcvtns {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtnd_u64_f64
uint64_t test_vcvtnd_u64_f64(float64_t a) {
  return vcvtnd_u64_f64(a);
// CHECK: fcvtnu {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtps_s32_f32
int32_t test_vcvtps_s32_f32(float32_t a) {
  return vcvtps_s32_f32(a);
// CHECK: fcvtps {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtps_u32_f32
uint32_t test_vcvtps_u32_f32(float32_t a) {
  return vcvtps_u32_f32(a);
// CHECK: fcvtpu {{w[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vcvtpd_s64_f64
int64_t test_vcvtpd_s64_f64(float64_t a) {
  return vcvtpd_s64_f64(a);
// CHECK: fcvtps {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtpd_u64_f64
uint64_t test_vcvtpd_u64_f64(float64_t a) {
  return vcvtpd_u64_f64(a);
// CHECK: fcvtpu {{x[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vcvtxd_f32_f64
float32_t test_vcvtxd_f32_f64(float64_t a) {
  return vcvtxd_f32_f64(a);
// CHECK: fcvtxn {{s[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vabds_f32
float32_t test_vabds_f32(float32_t a, float32_t b) {
  return vabds_f32(a, b);
  // CHECK: fabd {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vabdd_f64
float64_t test_vabdd_f64(float64_t a, float64_t b) {
  return vabdd_f64(a, b);
  // CHECK: fabd {{d[0-9]+}}, {{d[0-9]+}}
}

// CHECK: test_vmulxs_f32
float32_t test_vmulxs_f32(float32_t a, float32_t b) {
  return vmulxs_f32(a, b);
  // CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}
}

// CHECK: test_vmulxd_f64
float64_t test_vmulxd_f64(float64_t a, float64_t b) {
  return vmulxd_f64(a, b);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}
}
