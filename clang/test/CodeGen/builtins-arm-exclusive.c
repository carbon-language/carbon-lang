// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -Wall -Werror -triple thumbv7-linux-gnueabi -fno-signed-char -O3 -emit-llvm -o - %s | FileCheck %s

// Make sure the canonical use works before going into smaller details:
int atomic_inc(int *addr) {
  int Failure, OldVal;
  do {
    OldVal = __builtin_arm_ldrex(addr);
    Failure = __builtin_arm_strex(OldVal + 1, addr);
  } while (Failure);

  return OldVal;
}

// CHECK: @atomic_inc
// CHECK:   [[OLDVAL:%.*]] = tail call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
// CHECK:   [[INC:%.*]] = add nsw i32 [[OLDVAL]], 1
// CHECK:   [[FAILURE:%.*]] = tail call i32 @llvm.arm.strex.p0i32(i32 [[INC]], i32* %addr)
// CHECK:   [[TST:%.*]] = icmp eq i32 [[FAILURE]], 0
// CHECK:   br i1 [[TST]], label {{%[a-zA-Z0-9.]+}}, label {{%[a-zA-Z0-9.]+}}

struct Simple {
  char a, b;
};

int test_ldrex(char *addr, long long *addr64, float *addrfloat) {
// CHECK: @test_ldrex
  int sum = 0;
  sum += __builtin_arm_ldrex(addr);
// CHECK: [[INTRES:%.*]] = tail call i32 @llvm.arm.ldrex.p0i8(i8* %addr)
// CHECK: and i32 [[INTRES]], 255

  sum += __builtin_arm_ldrex((short *)addr);
// CHECK: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK: [[INTRES:%.*]] = tail call i32 @llvm.arm.ldrex.p0i16(i16* [[ADDR16]])
// CHECK: [[TMPSEXT:%.*]] = shl i32 [[INTRES]], 16
// CHECK: ashr exact i32 [[TMPSEXT]], 16

  sum += __builtin_arm_ldrex((int *)addr);
// CHECK: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK:  call i32 @llvm.arm.ldrex.p0i32(i32* [[ADDR32]])

  sum += __builtin_arm_ldrex((long long *)addr);
// CHECK: call { i32, i32 } @llvm.arm.ldrexd(i8* %addr)

  sum += __builtin_arm_ldrex(addr64);
// CHECK: [[ADDR64_AS8:%.*]] = bitcast i64* %addr64 to i8*
// CHECK: call { i32, i32 } @llvm.arm.ldrexd(i8* [[ADDR64_AS8]])

  sum += __builtin_arm_ldrex(addrfloat);
// CHECK: [[INTADDR:%.*]] = bitcast float* %addrfloat to i32*
// CHECK: [[INTRES:%.*]] = tail call i32 @llvm.arm.ldrex.p0i32(i32* [[INTADDR]])
// CHECK: bitcast i32 [[INTRES]] to float

  sum += __builtin_arm_ldrex((double *)addr);
// CHECK: [[STRUCTRES:%.*]] = tail call { i32, i32 } @llvm.arm.ldrexd(i8* %addr)
// CHECK: [[RESHI:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 1
// CHECK: [[RESLO:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 0
// CHECK: [[RESHI64:%.*]] = zext i32 [[RESHI]] to i64
// CHECK: [[RESLO64:%.*]] = zext i32 [[RESLO]] to i64
// CHECK: [[RESHIHI:%.*]] = shl nuw i64 [[RESHI64]], 32
// CHECK: [[INTRES:%.*]] = or i64 [[RESHIHI]], [[RESLO64]]
// CHECK: bitcast i64 [[INTRES]] to double

  sum += *__builtin_arm_ldrex((int **)addr);
// CHECK: [[INTRES:%.*]] = tail call i32 @llvm.arm.ldrex.p0i32(i32* [[ADDR32]])
// CHECK: inttoptr i32 [[INTRES]] to i32*

  sum += __builtin_arm_ldrex((struct Simple **)addr)->a;
// CHECK: [[INTRES:%.*]] = tail call i32 @llvm.arm.ldrex.p0i32(i32* [[ADDR32]])
// CHECK: inttoptr i32 [[INTRES]] to %struct.Simple*

  return sum;
}

int test_strex(char *addr) {
// CHECK: @test_strex
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_strex(4, addr);
// CHECK: call i32 @llvm.arm.strex.p0i8(i32 4, i8* %addr)

  res |= __builtin_arm_strex(42, (short *)addr);
// CHECK: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK:  call i32 @llvm.arm.strex.p0i16(i32 42, i16* [[ADDR16]])

  res |= __builtin_arm_strex(42, (int *)addr);
// CHECK: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK: call i32 @llvm.arm.strex.p0i32(i32 42, i32* [[ADDR32]])

  res |= __builtin_arm_strex(42, (long long *)addr);
// CHECK: call i32 @llvm.arm.strexd(i32 42, i32 0, i8* %addr)

  res |= __builtin_arm_strex(2.71828f, (float *)addr);
// CHECK: call i32 @llvm.arm.strex.p0i32(i32 1076754509, i32* [[ADDR32]])

  res |= __builtin_arm_strex(3.14159, (double *)addr);
// CHECK: call i32 @llvm.arm.strexd(i32 -266631570, i32 1074340345, i8* %addr)

  res |= __builtin_arm_strex(&var, (struct Simple **)addr);
// CHECK: [[INTVAL:%.*]] = ptrtoint i16* %var to i32
// CHECK: call i32 @llvm.arm.strex.p0i32(i32 [[INTVAL]], i32* [[ADDR32]])

  return res;
}

void test_clrex() {
// CHECK: @test_clrex

  __builtin_arm_clrex();
// CHECK: call void @llvm.arm.clrex()
}
