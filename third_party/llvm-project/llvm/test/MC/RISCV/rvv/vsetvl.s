# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsetvli a2, a0, e32, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e32,  m1,  ta,  ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 05 0d <unknown>

vsetvli a2, a0, e32, m2, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m2, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x15,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 15 0d <unknown>

vsetvli a2, a0, e32, m4, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m4, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x25,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 25 0d <unknown>

vsetvli a2, a0, e32, m8, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m8, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x35,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 35 0d <unknown>

vsetvli a2, a0, e32, mf2, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, mf2, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x75,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 75 0d <unknown>

vsetvli a2, a0, e32, mf4, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, mf4, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x65,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 65 0d <unknown>

vsetvli a2, a0, e32, mf8, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, mf8, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x55,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 55 0d <unknown>

vsetvli a2, a0, e32, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x0d]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 05 0d <unknown>

vsetvli a2, a0, e32, m1, tu, ma
# CHECK-INST: vsetvli a2, a0, e32, m1, tu, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x09]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 05 09 <unknown>

vsetvli a2, a0, e32, m1, ta, mu
# CHECK-INST: vsetvli a2, a0, e32, m1, ta, mu
# CHECK-ENCODING: [0x57,0x76,0x05,0x05]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 05 05 <unknown>

vsetvli a2, a0, e32, m1, tu, mu
# CHECK-INST: vsetvli a2, a0, e32, m1
# CHECK-ENCODING: [0x57,0x76,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 05 01 <unknown>

vsetvl a2, a0, a1
# CHECK-INST: vsetvl a2, a0, a1
# CHECK-ENCODING: [0x57,0x76,0xb5,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 b5 80 <unknown>

vsetivli a2, 0, e32, m1, ta, ma
# CHECK-INST: vsetivli a2, 0, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x00,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 76 00 cd <unknown>

vsetivli a2, 15, e32, m1, ta, ma
# CHECK-INST: vsetivli a2, 15, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x07,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 f6 07 cd <unknown>

vsetivli a2, 31, e32, m1, ta, ma
# CHECK-INST: vsetivli a2, 31, e32, m1, ta, ma
# CHECK-ENCODING: [0x57,0xf6,0x0f,0xcd]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 f6 0f cd <unknown>
