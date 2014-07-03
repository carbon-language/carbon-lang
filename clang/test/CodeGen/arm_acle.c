// RUN: %clang_cc1 -ffreestanding -triple armv8 -target-cpu cortex-a57 -O -S -emit-llvm -o - %s | FileCheck %s -check-prefix=ARM -check-prefix=AArch32
// RUN: %clang_cc1 -ffreestanding -triple aarch64 -target-cpu cortex-a57 -O -S -emit-llvm -o - %s | FileCheck %s -check-prefix=ARM -check-prefix=AArch64

#include <arm_acle.h>

/* Miscellaneous data-processing intrinsics */
// ARM-LABEL: test_rev
// ARM: call i32 @llvm.bswap.i32(i32 %t)
uint32_t test_rev(uint32_t t) {
  return __rev(t);
}

// ARM-LABEL: test_revl
// AArch32: call i32 @llvm.bswap.i32(i32 %t)
// AArch64: call i64 @llvm.bswap.i64(i64 %t)
long test_revl(long t) {
  return __revl(t);
}

// ARM-LABEL: test_revll
// ARM: call i64 @llvm.bswap.i64(i64 %t)
uint64_t test_revll(uint64_t t) {
  return __revll(t);
}

// ARM-LABEL: test_clz
// ARM: call i32 @llvm.ctlz.i32(i32 %t, i1 false)
uint32_t test_clz(uint32_t t) {
  return __clz(t);
}

// ARM-LABEL: test_clzl
// AArch32: call i32 @llvm.ctlz.i32(i32 %t, i1 false)
// AArch64: call i64 @llvm.ctlz.i64(i64 %t, i1 false)
long test_clzl(long t) {
  return __clzl(t);
}

// ARM-LABEL: test_clzll
// ARM: call i64 @llvm.ctlz.i64(i64 %t, i1 false)
uint64_t test_clzll(uint64_t t) {
  return __clzll(t);
}

/* Saturating intrinsics */
#ifdef __ARM_32BIT_STATE
// AArch32-LABEL: test_ssat
// AArch32: call i32 @llvm.arm.ssat(i32 %t, i32 1)
int32_t test_ssat(int32_t t) {
  return __ssat(t, 1);
}

// AArch32-LABEL: test_usat
// AArch32: call i32 @llvm.arm.usat(i32 %t, i32 2)
int32_t test_usat(int32_t t) {
  return __usat(t, 2);
}
// AArch32-LABEL: test_qadd
// AArch32: call i32 @llvm.arm.qadd(i32 %a, i32 %b)
int32_t test_qadd(int32_t a, int32_t b) {
  return __qadd(a, b);
}

// AArch32-LABEL: test_qsub
// AArch32: call i32 @llvm.arm.qsub(i32 %a, i32 %b)
int32_t test_qsub(int32_t a, int32_t b) {
  return __qsub(a, b);
}

extern int32_t f();
// AArch32-LABEL: test_qdbl
// AArch32: [[VAR:%[a-z0-9]+]] = {{.*}} call {{.*}} @f
// AArch32-NOT: call {{.*}} @f
// AArch32: call i32 @llvm.arm.qadd(i32 [[VAR]], i32 [[VAR]])
int32_t test_qdbl() {
  return __qdbl(f());
}
#endif

/* CRC32 intrinsics */
// ARM-LABEL: test_crc32b
// AArch32: call i32 @llvm.arm.crc32b
// AArch64: call i32 @llvm.aarch64.crc32b
uint32_t test_crc32b(uint32_t a, uint8_t b) {
  return __crc32b(a, b);
}

// ARM-LABEL: test_crc32h
// AArch32: call i32 @llvm.arm.crc32h
// AArch64: call i32 @llvm.aarch64.crc32h
uint32_t test_crc32h(uint32_t a, uint16_t b) {
  return __crc32h(a, b);
}

// ARM-LABEL: test_crc32w
// AArch32: call i32 @llvm.arm.crc32w
// AArch64: call i32 @llvm.aarch64.crc32w
uint32_t test_crc32w(uint32_t a, uint32_t b) {
  return __crc32w(a, b);
}

// ARM-LABEL: test_crc32d
// AArch32: call i32 @llvm.arm.crc32w
// AArch32: call i32 @llvm.arm.crc32w
// AArch64: call i32 @llvm.aarch64.crc32x
uint32_t test_crc32d(uint32_t a, uint64_t b) {
  return __crc32d(a, b);
}

// ARM-LABEL: test_crc32cb
// AArch32: call i32 @llvm.arm.crc32cb
// AArch64: call i32 @llvm.aarch64.crc32cb
uint32_t test_crc32cb(uint32_t a, uint8_t b) {
  return __crc32cb(a, b);
}

// ARM-LABEL: test_crc32ch
// AArch32: call i32 @llvm.arm.crc32ch
// AArch64: call i32 @llvm.aarch64.crc32ch
uint32_t test_crc32ch(uint32_t a, uint16_t b) {
  return __crc32ch(a, b);
}

// ARM-LABEL: test_crc32cw
// AArch32: call i32 @llvm.arm.crc32cw
// AArch64: call i32 @llvm.aarch64.crc32cw
uint32_t test_crc32cw(uint32_t a, uint32_t b) {
  return __crc32cw(a, b);
}

// ARM-LABEL: test_crc32cd
// AArch32: call i32 @llvm.arm.crc32cw
// AArch32: call i32 @llvm.arm.crc32cw
// AArch64: call i32 @llvm.aarch64.crc32cx
uint32_t test_crc32cd(uint32_t a, uint64_t b) {
  return __crc32cd(a, b);
}
