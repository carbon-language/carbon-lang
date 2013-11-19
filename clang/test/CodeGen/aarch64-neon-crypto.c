// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -target-feature +crypto -S -O3 -o - %s | FileCheck %s
// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -S -O3 -o - %s 2>&1 | FileCheck --check-prefix=CHECK-NO-CRYPTO %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

uint8x16_t test_vaeseq_u8(uint8x16_t data, uint8x16_t key) {
  // CHECK: test_vaeseq_u8
  // CHECK-NO-CRYPTO: warning: implicit declaration of function 'vaeseq_u8' is invalid in C99
  return vaeseq_u8(data, key);
  // CHECK: aese {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x16_t test_vaesdq_u8(uint8x16_t data, uint8x16_t key) {
  // CHECK: test_vaesdq_u8
  return vaesdq_u8(data, key);
  // CHECK: aesd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x16_t test_vaesmcq_u8(uint8x16_t data) {
  // CHECK: test_vaesmcq_u8
  return vaesmcq_u8(data);
  // CHECK: aesmc {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x16_t test_vaesimcq_u8(uint8x16_t data) {
  // CHECK: test_vaesimcq_u8
  return vaesimcq_u8(data);
  // CHECK: aesimc {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32_t test_vsha1h_u32(uint32_t hash_e) {
  // CHECK: test_vsha1h_u32
  return vsha1h_u32(hash_e);
  // CHECK: sha1h {{s[0-9]+}}, {{s[0-9]+}}
}

uint32x4_t test_vsha1su1q_u32(uint32x4_t tw0_3, uint32x4_t w12_15) {
  // CHECK: test_vsha1su1q_u32
  return vsha1su1q_u32(tw0_3, w12_15);
  // CHECK: sha1su1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha256su0q_u32(uint32x4_t w0_3, uint32x4_t w4_7) {
  // CHECK: test_vsha256su0q_u32
  return vsha256su0q_u32(w0_3, w4_7);
  // CHECK: sha256su0 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha1cq_u32(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK: test_vsha1cq_u32
  return vsha1cq_u32(hash_abcd, hash_e, wk);
  // CHECK: sha1c {{q[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha1pq_u32(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK: test_vsha1pq_u32
  return vsha1pq_u32(hash_abcd, hash_e, wk);
  // CHECK: sha1p {{q[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha1mq_u32(uint32x4_t hash_abcd, uint32_t hash_e, uint32x4_t wk) {
  // CHECK: test_vsha1mq_u32
  return vsha1mq_u32(hash_abcd, hash_e, wk);
  // CHECK: sha1m {{q[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha1su0q_u32(uint32x4_t w0_3, uint32x4_t w4_7, uint32x4_t w8_11) {
  // CHECK: test_vsha1su0q_u32
  return vsha1su0q_u32(w0_3, w4_7, w8_11);
  // CHECK: sha1su0 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha256hq_u32(uint32x4_t hash_abcd, uint32x4_t hash_efgh, uint32x4_t wk) {
  // CHECK: test_vsha256hq_u32
  return vsha256hq_u32(hash_abcd, hash_efgh, wk);
  // CHECK: sha256h {{q[0-9]+}}, {{q[0-9]+}}, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha256h2q_u32(uint32x4_t hash_efgh, uint32x4_t hash_abcd, uint32x4_t wk) {
  // CHECK: test_vsha256h2q_u32
  return vsha256h2q_u32(hash_efgh, hash_abcd, wk);
  // CHECK: sha256h2 {{q[0-9]+}}, {{q[0-9]+}}, {{v[0-9]+}}.4s
}

uint32x4_t test_vsha256su1q_u32(uint32x4_t tw0_3, uint32x4_t w8_11, uint32x4_t w12_15) {
  // CHECK: test_vsha256su1q_u32
  return vsha256su1q_u32(tw0_3, w8_11, w12_15);
  // CHECK: sha256su1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
