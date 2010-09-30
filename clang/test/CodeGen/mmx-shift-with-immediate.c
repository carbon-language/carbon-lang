// RUN: %clang -mmmx -ccc-host-triple i386-unknown-unknown -emit-llvm -S %s -o - | FileCheck %s
#include <mmintrin.h>

void shift(__m64 a, __m64 b, int c) {
  // CHECK: x86_mmx @llvm.x86.mmx.pslli.w(x86_mmx %{{.*}}, i32 {{.*}})
  _mm_slli_pi16(a, c);
  // CHECK: x86_mmx @llvm.x86.mmx.pslli.d(x86_mmx %{{.*}}, i32 {{.*}})
  _mm_slli_pi32(a, c);
  // FIXME: <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64> %{{.*}}, i32 {{.*}})
  // This is currently lowered into non-intrinsic instructions. This may not be
  // correct once the MMX reworking is finished.
  _mm_slli_si64(a, c);

  // CHECK: x86_mmx @llvm.x86.mmx.psrli.w(x86_mmx %{{.*}}, i32 {{.*}})
  _mm_srli_pi16(a, c);
  // CHECK: x86_mmx @llvm.x86.mmx.psrli.d(x86_mmx %{{.*}}, i32 {{.*}})
  _mm_srli_pi32(a, c);
  // FIXME: <1 x i64> @llvm.x86.mmx.psrli.q(<1 x i64> %{{.*}}, i32 {{.*}})
  // See above.
  _mm_srli_si64(a, c);

  // CHECK: x86_mmx @llvm.x86.mmx.psrai.w(x86_mmx %{{.*}}, i32 {{.*}})
  _mm_srai_pi16(a, c);
  // CHECK: x86_mmx @llvm.x86.mmx.psrai.d(x86_mmx %{{.*}}, i32 {{.*}})
  _mm_srai_pi32(a, c);
}
