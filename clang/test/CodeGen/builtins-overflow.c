// Test CodeGen for Security Check Overflow Builtins.
// rdar://13421498

// RUN: %clang_cc1 -triple "i686-unknown-unknown"   -emit-llvm -x c %s -o - -O0 | FileCheck %s
// RUN: %clang_cc1 -triple "x86_64-unknown-unknown" -emit-llvm -x c %s -o - -O0 | FileCheck %s
// RUN: %clang_cc1 -triple "x86_64-mingw32"         -emit-llvm -x c %s -o - -O0 | FileCheck %s

extern unsigned UnsignedErrorCode;
extern unsigned long UnsignedLongErrorCode;
extern unsigned long long UnsignedLongLongErrorCode;
extern int IntErrorCode;
extern long LongErrorCode;
extern long long LongLongErrorCode;

unsigned test_uadd_overflow(unsigned x, unsigned y) {
// CHECK: @test_uadd_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  unsigned result;
  if (__builtin_uadd_overflow(x, y, &result))
    return UnsignedErrorCode;
  return result;
}

unsigned long test_uaddl_overflow(unsigned long x, unsigned long y) {
// CHECK: @test_uaddl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.uadd.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  unsigned long result;
  if (__builtin_uaddl_overflow(x, y, &result))
    return UnsignedLongErrorCode;
  return result;
}

unsigned long long test_uaddll_overflow(unsigned long long x, unsigned long long y) {
// CHECK: @test_uaddll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  unsigned long long result;
  if (__builtin_uaddll_overflow(x, y, &result))
    return UnsignedLongLongErrorCode;
  return result;
}

unsigned test_usub_overflow(unsigned x, unsigned y) {
// CHECK: @test_usub_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  unsigned result;
  if (__builtin_usub_overflow(x, y, &result))
    return UnsignedErrorCode;
  return result;
}

unsigned long test_usubl_overflow(unsigned long x, unsigned long y) {
// CHECK: @test_usubl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.usub.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  unsigned long result;
  if (__builtin_usubl_overflow(x, y, &result))
    return UnsignedLongErrorCode;
  return result;
}

unsigned long long test_usubll_overflow(unsigned long long x, unsigned long long y) {
// CHECK: @test_usubll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  unsigned long long result;
  if (__builtin_usubll_overflow(x, y, &result))
    return UnsignedLongLongErrorCode;
  return result;
}

unsigned test_umul_overflow(unsigned x, unsigned y) {
// CHECK: @test_umul_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  unsigned result;
  if (__builtin_umul_overflow(x, y, &result))
    return UnsignedErrorCode;
  return result;
}

unsigned long test_umull_overflow(unsigned long x, unsigned long y) {
// CHECK: @test_umull_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.umul.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  unsigned long result;
  if (__builtin_umull_overflow(x, y, &result))
    return UnsignedLongErrorCode;
  return result;
}

unsigned long long test_umulll_overflow(unsigned long long x, unsigned long long y) {
// CHECK: @test_umulll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  unsigned long long result;
  if (__builtin_umulll_overflow(x, y, &result))
    return UnsignedLongLongErrorCode;
  return result;
}

int test_sadd_overflow(int x, int y) {
// CHECK: @test_sadd_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  int result;
  if (__builtin_sadd_overflow(x, y, &result))
    return IntErrorCode;
  return result;
}

long test_saddl_overflow(long x, long y) {
// CHECK: @test_saddl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.sadd.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  long result;
  if (__builtin_saddl_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

long long test_saddll_overflow(long long x, long long y) {
// CHECK: @test_saddll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  long long result;
  if (__builtin_saddll_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}

int test_ssub_overflow(int x, int y) {
// CHECK: @test_ssub_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  int result;
  if (__builtin_ssub_overflow(x, y, &result))
    return IntErrorCode;
  return result;
}

long test_ssubl_overflow(long x, long y) {
// CHECK: @test_ssubl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.ssub.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  long result;
  if (__builtin_ssubl_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

long long test_ssubll_overflow(long long x, long long y) {
// CHECK: @test_ssubll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  long long result;
  if (__builtin_ssubll_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}

int test_smul_overflow(int x, int y) {
// CHECK: @test_smul_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  int result;
  if (__builtin_smul_overflow(x, y, &result))
    return IntErrorCode;
  return result;
}

long test_smull_overflow(long x, long y) {
// CHECK: @test_smull_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.smul.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  long result;
  if (__builtin_smull_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

long long test_smulll_overflow(long long x, long long y) {
// CHECK: @test_smulll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  long long result;
  if (__builtin_smulll_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}
