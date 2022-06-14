// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv8m.base-none-eabi -O1 -emit-llvm %s -o - | FileCheck %s
int test_cmse_TT(void *p){
  return __builtin_arm_cmse_TT(p);
  // CHECK: call i32 @llvm.arm.cmse.tt(i8* %{{.*}})
}

int test_cmse_TTT(void *p){
  return __builtin_arm_cmse_TTT(p);
  // CHECK: call i32 @llvm.arm.cmse.ttt(i8* %{{.*}})
}

int test_cmse_TTA(void *p){
  return __builtin_arm_cmse_TTA(p);
  // CHECK: call i32 @llvm.arm.cmse.tta(i8* %{{.*}})
}

int test_cmse_TTAT(void *p){
  return __builtin_arm_cmse_TTAT(p);
  // CHECK: call i32 @llvm.arm.cmse.ttat(i8* %{{.*}})
}
