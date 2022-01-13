// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W64-ERR --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// ENC_VOPC, VOP3 variant.
//===----------------------------------------------------------------------===//

v_cmp_f_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x00,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x00,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x00,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x00,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x01,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x01,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x01,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x01,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x02,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x02,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x02,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x02,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x03,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x03,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x03,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x03,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x04,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x04,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x04,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x04,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x05,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x05,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x05,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x05,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x06,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x06,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x06,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x06,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x07,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x07,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x07,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x07,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x08,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x08,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x08,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x08,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x09,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x09,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x09,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x09,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x0a,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x0a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x0b,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x0b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x0c,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x0c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x0d,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x0d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x0e,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x0e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0x0f,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f32_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0x0f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x20,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x20,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x20,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x20,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x21,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x21,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x21,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x21,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x22,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x22,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x22,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x22,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x23,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x23,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x23,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x23,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x24,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x24,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x24,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x24,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x25,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x25,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x25,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x25,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x26,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x26,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x26,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x26,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x27,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x27,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x27,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x27,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x28,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x28,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x28,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x28,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x29,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x29,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x29,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x29,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2a,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x2a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2b,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x2b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2c,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x2c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2d,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x2d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2e,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x2e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], -v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], -v[1:2], -v[2:3]
// W64: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s[10:11], v[1:2], v[2:3] clamp
// W64: encoding: [0x0a,0x80,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, -v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, -v[1:2], -v[2:3]
// W32: encoding: [0x0a,0x00,0x2f,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f64_e64 s10, v[1:2], v[2:3] clamp
// W32: encoding: [0x0a,0x80,0x2f,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x80,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x80,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x81,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x81,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x82,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x82,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x83,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x83,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x84,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x84,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x85,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x85,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x86,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x86,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x87,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x87,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x88,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f32_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0x88,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x89,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x89,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x8a,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x8a,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x8b,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x8b,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x8c,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x8c,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x8d,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x8d,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0x8e,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0x8e,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0x8f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0x8f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0x8f,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0x8f,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[12:13], v[1:2], v2
// W64: encoding: [0x0c,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[100:101], v[1:2], v2
// W64: encoding: [0x64,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 vcc, v[1:2], v2
// W64: encoding: [0x6a,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[254:255], v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], s[2:3], v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], s[4:5], v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], s[100:101], v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], vcc, v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], exec, v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], v255
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], s2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], s101
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], vcc_lo
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], vcc_hi
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], m0
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], exec_lo
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], exec_hi
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s[10:11], -v[1:2], v2
// W64: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[12:13], v[1:2], v[2:3]
// W64: encoding: [0x0c,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[100:101], v[1:2], v[2:3]
// W64: encoding: [0x64,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 vcc, v[1:2], v[2:3]
// W64: encoding: [0x6a,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[254:255], v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0xfe,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], s[2:3], v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x02,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], s[4:5], v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x04,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], s[100:101], v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x64,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], vcc, v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], exec, v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], 0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], -1, v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], 0.5, v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], -4.0, v[2:3]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], v[254:255]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xfd,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], s[4:5]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x09,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], s[6:7]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x0d,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], s[100:101]
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xc9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], vcc
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], exec
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], 0
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], -1
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], 0.5
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s[10:11], v[1:2], -4.0
// W64: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa0,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa1,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa2,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa3,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa4,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa5,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa6,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xa7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_i64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa7,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s12, v[1:2], v2
// W32: encoding: [0x0c,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s100, v[1:2], v2
// W32: encoding: [0x64,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 vcc_lo, v[1:2], v2
// W32: encoding: [0x6a,0x00,0xa8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[254:255], v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, s[2:3], v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, s[4:5], v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, s[100:101], v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, vcc, v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, exec, v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], v255
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], s2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], s101
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], vcc_lo
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], vcc_hi
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], m0
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], exec_lo
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], exec_hi
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_class_f64_e64 s10, -v[1:2], v2
// W32: encoding: [0x0a,0x00,0xa8,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe0,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe1,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe2,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe3,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe4,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe5,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe6,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s12, v[1:2], v[2:3]
// W32: encoding: [0x0c,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s100, v[1:2], v[2:3]
// W32: encoding: [0x64,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 vcc_lo, v[1:2], v[2:3]
// W32: encoding: [0x6a,0x00,0xe7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[254:255], v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0xfe,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, s[2:3], v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x02,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, s[4:5], v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x04,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, s[100:101], v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x64,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, vcc, v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, exec, v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, 0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, -1, v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, 0.5, v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, -4.0, v[2:3]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], v[254:255]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xfd,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], s[4:5]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x09,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], s[6:7]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x0d,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], s[100:101]
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xc9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], vcc
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], exec
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], 0
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], -1
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], 0.5
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u64_e64 s10, v[1:2], -4.0
// W32: encoding: [0x0a,0x00,0xe7,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xca,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xce,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xea,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xec,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xed,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xee,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[12:13], v1, v2
// W64: encoding: [0x0c,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[100:101], v1, v2
// W64: encoding: [0x64,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 vcc, v1, v2
// W64: encoding: [0x6a,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v255, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0xff,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], s1, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], s101, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x65,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], vcc_lo, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x6a,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], vcc_hi, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x6b,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], m0, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x7c,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], exec_lo, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x7e,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], exec_hi, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x7f,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], 0, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x80,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], -1, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0xc1,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], 0.5, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0xf0,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], -4.0, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0xf7,0x04,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, v255
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xff,0x03,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, s2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, s101
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xcb,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, vcc_lo
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xd5,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, vcc_hi
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xd7,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, m0
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xf9,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, exec_lo
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xfd,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, exec_hi
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xff,0x00,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, 0
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x01,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, -1
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x83,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, 0.5
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xe1,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, -4.0
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xef,0x01,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], -v1, v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x20]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, -v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x40]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], -v1, -v2
// W64: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x60]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s[10:11], v1, v2 clamp
// W64: encoding: [0x0a,0x80,0xef,0xd4,0x01,0x05,0x02,0x00]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xa9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xa9,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xaa,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xaa,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xab,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xab,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xac,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xac,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xad,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xad,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xae,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0xff,0x04,0x02,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x01,0x00,0x00,0x38,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xae,0xd4,0x01,0xff,0x01,0x00,0x00,0xc4,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc0,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc0,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc1,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc1,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc2,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc2,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc3,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc3,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc4,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc4,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc5,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ne_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc5,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc6,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc6,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc7,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_t_u32_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc7,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xc8,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_f_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xc8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xc9,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lt_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xc9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xca,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xca,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_eq_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xca,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xcb,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_le_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xcb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xcc,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_gt_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xcc,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xcd,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_lg_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xcd,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xce,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xce,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ge_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xce,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xcf,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_o_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xcf,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xe8,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_u_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xe8,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xe9,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nge_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xe9,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xea,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xea,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlg_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xea,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xeb,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_ngt_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xeb,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xec,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xec,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nle_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xec,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xed,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xed,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_neq_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xed,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xee,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xee,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_nlt_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xee,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s12, v1, v2
// W32: encoding: [0x0c,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s100, v1, v2
// W32: encoding: [0x64,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 vcc_lo, v1, v2
// W32: encoding: [0x6a,0x00,0xef,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v255, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0xff,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, s1, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, s101, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x65,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, vcc_lo, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x6a,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, vcc_hi, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x6b,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, m0, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x7c,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, exec_lo, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x7e,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, exec_hi, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x7f,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, 0, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x80,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, -1, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0xc1,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, 0.5, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0xf0,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, -4.0, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0xf7,0x04,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, v255
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xff,0x03,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, s2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, s101
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xcb,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, vcc_lo
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xd5,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, vcc_hi
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xd7,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, m0
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xf9,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, exec_lo
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xfd,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, exec_hi
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xff,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, 0
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x01,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, -1
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x83,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, 0.5
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xe1,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, -4.0
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0xef,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, -v1, v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x20]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, -v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, -v1, -v2
// W32: encoding: [0x0a,0x00,0xef,0xd4,0x01,0x05,0x02,0x60]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmp_tru_f16_e64 s10, v1, v2 clamp
// W32: encoding: [0x0a,0x80,0xef,0xd4,0x01,0x05,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
