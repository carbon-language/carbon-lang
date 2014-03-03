// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x8_t test_vtbl1_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtbl1_s8
  return vtbl1_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbl1_s8(int8x16_t a, int8x8_t b) {
  // CHECK-LABEL: test_vqtbl1_s8
  return vqtbl1_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vtbl2_s8(int8x8x2_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtbl2_s8
  return vtbl2_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbl2_s8(int8x16x2_t a, int8x8_t b) {
  // CHECK-LABEL: test_vqtbl2_s8
  return vqtbl2_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vtbl3_s8(int8x8x3_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtbl3_s8
  return vtbl3_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbl3_s8(int8x16x3_t a, int8x8_t b) {
  // CHECK-LABEL: test_vqtbl3_s8
  return vqtbl3_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vtbl4_s8(int8x8x4_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtbl4_s8
  return vtbl4_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbl4_s8(int8x16x4_t a, int8x8_t b) {
  // CHECK-LABEL: test_vqtbl4_s8
  return vqtbl4_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x16_t test_vqtbl1q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vqtbl1q_s8
  return vqtbl1q_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x16_t test_vqtbl2q_s8(int8x16x2_t a, int8x16_t b) {
  // CHECK-LABEL: test_vqtbl2q_s8
  return vqtbl2q_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x16_t test_vqtbl3q_s8(int8x16x3_t a, int8x16_t b) {
  // CHECK-LABEL: test_vqtbl3q_s8
  return vqtbl3q_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x16_t test_vqtbl4q_s8(int8x16x4_t a, int8x16_t b) {
  // CHECK-LABEL: test_vqtbl4q_s8
  return vqtbl4q_s8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x8_t test_vtbx1_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
  // CHECK-LABEL: test_vtbx1_s8
  return vtbx1_s8(a, b, c);
  // CHECK: movi {{v[0-9]+}}.8b, #0
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
  // CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vtbx2_s8(int8x8_t a, int8x8x2_t b, int8x8_t c) {
  // CHECK-LABEL: test_vtbx2_s8
  return vtbx2_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vtbx3_s8(int8x8_t a, int8x8x3_t b, int8x8_t c) {
  // CHECK-LABEL: test_vtbx3_s8
  return vtbx3_s8(a, b, c);
  // CHECK: movi {{v[0-9]+}}.8b, #0
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
  // CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vtbx4_s8(int8x8_t a, int8x8x4_t b, int8x8_t c) {
  // CHECK-LABEL: test_vtbx4_s8
  return vtbx4_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbx1_s8(int8x8_t a, int8x16_t b, int8x8_t c) {
  // CHECK-LABEL: test_vqtbx1_s8
  return vqtbx1_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbx2_s8(int8x8_t a, int8x16x2_t b, int8x8_t c) {
  // CHECK-LABEL: test_vqtbx2_s8
  return vqtbx2_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbx3_s8(int8x8_t a, int8x16x3_t b, int8x8_t c) {
  // CHECK-LABEL: test_vqtbx3_s8
  return vqtbx3_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x8_t test_vqtbx4_s8(int8x8_t a, int8x16x4_t b, int8x8_t c) {
  // CHECK-LABEL: test_vqtbx4_s8
  return vqtbx4_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

int8x16_t test_vqtbx1q_s8(int8x16_t a, int8x16_t b, int8x16_t c) {
  // CHECK-LABEL: test_vqtbx1q_s8
  return vqtbx1q_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x16_t test_vqtbx2q_s8(int8x16_t a, int8x16x2_t b, int8x16_t c) {
  // CHECK-LABEL: test_vqtbx2q_s8
  return vqtbx2q_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x16_t test_vqtbx3q_s8(int8x16_t a, int8x16x3_t b, int8x16_t c) {
  // CHECK-LABEL: test_vqtbx3q_s8
  return vqtbx3q_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

int8x16_t test_vqtbx4q_s8(int8x16_t a, int8x16x4_t b, int8x16_t c) {
  // CHECK-LABEL: test_vqtbx4q_s8
  return vqtbx4q_s8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x8_t test_vtbl1_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl1_u8
  return vtbl1_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbl1_u8(uint8x16_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl1_u8
  return vqtbl1_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vtbl2_u8(uint8x8x2_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl2_u8
  return vtbl2_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbl2_u8(uint8x16x2_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl2_u8
  return vqtbl2_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vtbl3_u8(uint8x8x3_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl3_u8
  return vtbl3_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbl3_u8(uint8x16x3_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl3_u8
  return vqtbl3_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vtbl4_u8(uint8x8x4_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl4_u8
  return vtbl4_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbl4_u8(uint8x16x4_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl4_u8
  return vqtbl4_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x16_t test_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl1q_u8
  return vqtbl1q_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x16_t test_vqtbl2q_u8(uint8x16x2_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl2q_u8
  return vqtbl2q_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x16_t test_vqtbl3q_u8(uint8x16x3_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl3q_u8
  return vqtbl3q_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x16_t test_vqtbl4q_u8(uint8x16x4_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl4q_u8
  return vqtbl4q_u8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x8_t test_vtbx1_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx1_u8
  return vtbx1_u8(a, b, c);
  // CHECK: movi {{v[0-9]+}}.8b, #0
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
  // CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x8_t test_vtbx2_u8(uint8x8_t a, uint8x8x2_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx2_u8
  return vtbx2_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vtbx3_u8(uint8x8_t a, uint8x8x3_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx3_u8
  return vtbx3_u8(a, b, c);
  // CHECK: movi {{v[0-9]+}}.8b, #0
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
  // CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x8_t test_vtbx4_u8(uint8x8_t a, uint8x8x4_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx4_u8
  return vtbx4_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbx1_u8(uint8x8_t a, uint8x16_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx1_u8
  return vqtbx1_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbx2_u8(uint8x8_t a, uint8x16x2_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx2_u8
  return vqtbx2_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbx3_u8(uint8x8_t a, uint8x16x3_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx3_u8
  return vqtbx3_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x8_t test_vqtbx4_u8(uint8x8_t a, uint8x16x4_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx4_u8
  return vqtbx4_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

uint8x16_t test_vqtbx1q_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx1q_u8
  return vqtbx1q_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x16_t test_vqtbx2q_u8(uint8x16_t a, uint8x16x2_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx2q_u8
  return vqtbx2q_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x16_t test_vqtbx3q_u8(uint8x16_t a, uint8x16x3_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx3q_u8
  return vqtbx3q_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

uint8x16_t test_vqtbx4q_u8(uint8x16_t a, uint8x16x4_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx4q_u8
  return vqtbx4q_u8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x8_t test_vtbl1_p8(poly8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl1_p8
  return vtbl1_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbl1_p8(poly8x16_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl1_p8
  return vqtbl1_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vtbl2_p8(poly8x8x2_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl2_p8
  return vtbl2_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbl2_p8(poly8x16x2_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl2_p8
  return vqtbl2_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vtbl3_p8(poly8x8x3_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl3_p8
  return vtbl3_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbl3_p8(poly8x16x3_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl3_p8
  return vqtbl3_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vtbl4_p8(poly8x8x4_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtbl4_p8
  return vtbl4_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbl4_p8(poly8x16x4_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vqtbl4_p8
  return vqtbl4_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x16_t test_vqtbl1q_p8(poly8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl1q_p8
  return vqtbl1q_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x16_t test_vqtbl2q_p8(poly8x16x2_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl2q_p8
  return vqtbl2q_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x16_t test_vqtbl3q_p8(poly8x16x3_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl3q_p8
  return vqtbl3q_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x16_t test_vqtbl4q_p8(poly8x16x4_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vqtbl4q_p8
  return vqtbl4q_p8(a, b);
  // CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x8_t test_vtbx1_p8(poly8x8_t a, poly8x8_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx1_p8
  return vtbx1_p8(a, b, c);
  // CHECK: movi {{v[0-9]+}}.8b, #0
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
  // CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x8_t test_vtbx2_p8(poly8x8_t a, poly8x8x2_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx2_p8
  return vtbx2_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vtbx3_p8(poly8x8_t a, poly8x8x3_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx3_p8
  return vtbx3_p8(a, b, c);
  // CHECK: movi {{v[0-9]+}}.8b, #0
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
  // CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
  // CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x8_t test_vtbx4_p8(poly8x8_t a, poly8x8x4_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vtbx4_p8
  return vtbx4_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbx1_p8(poly8x8_t a, uint8x16_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx1_p8
  return vqtbx1_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbx2_p8(poly8x8_t a, poly8x16x2_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx2_p8
  return vqtbx2_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbx3_p8(poly8x8_t a, poly8x16x3_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx3_p8
  return vqtbx3_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x8_t test_vqtbx4_p8(poly8x8_t a, poly8x16x4_t b, uint8x8_t c) {
  // CHECK-LABEL: test_vqtbx4_p8
  return vqtbx4_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
}

poly8x16_t test_vqtbx1q_p8(poly8x16_t a, uint8x16_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx1q_p8
  return vqtbx1q_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x16_t test_vqtbx2q_p8(poly8x16_t a, poly8x16x2_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx2q_p8
  return vqtbx2q_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x16_t test_vqtbx3q_p8(poly8x16_t a, poly8x16x3_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx3q_p8
  return vqtbx3q_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}

poly8x16_t test_vqtbx4q_p8(poly8x16_t a, poly8x16x4_t b, uint8x16_t c) {
  // CHECK-LABEL: test_vqtbx4q_p8
  return vqtbx4q_p8(a, b, c);
  // CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
}
