// RUN: %clang_cc1 -fmath-errno -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefix=HAS_ERRNO
// RUN: %clang_cc1              -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefix=NO_ERRNO

float foo(float X) {
  // HAS_ERRNO: call float @sqrtf(float
  // NO_ERRNO: call float @llvm.sqrt.f32(float
  return __builtin_sqrtf(X);
}

// HAS_ERRNO: declare float @sqrtf(float) [[ATTR:#[0-9]+]]
// HAS_ERRNO-NOT: attributes [[ATTR]] = {{{.*}} readnone

// NO_ERRNO: declare float @llvm.sqrt.f32(float) [[ATTR:#[0-9]+]]
// NO_ERRNO: attributes [[ATTR]] = { nofree nosync nounwind readnone {{.*}}}

