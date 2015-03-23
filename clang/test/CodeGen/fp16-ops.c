// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -emit-llvm -o - -triple arm-none-linux-gnueabi %s | FileCheck %s --check-prefix=NOHALF --check-prefix=CHECK
// RUN: %clang_cc1 -emit-llvm -o - -triple aarch64-none-linux-gnueabi %s | FileCheck %s --check-prefix=NOHALF --check-prefix=CHECK
// RUN: %clang_cc1 -emit-llvm -o - -triple arm-none-linux-gnueabi -fallow-half-arguments-and-returns %s | FileCheck %s --check-prefix=HALF --check-prefix=CHECK
// RUN: %clang_cc1 -emit-llvm -o - -triple aarch64-none-linux-gnueabi -fallow-half-arguments-and-returns %s | FileCheck %s --check-prefix=HALF --check-prefix=CHECK
typedef unsigned cond_t;

volatile cond_t test;
volatile __fp16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;

void foo(void) {
  // CHECK-LABEL: define void @foo()

  // Check unary ops

  // NOHALF: [[F16TOF32:call float @llvm.convert.from.fp16.f32]]
  // HALF: [[F16TOF32:fpext half]]
  // CHECK: fptoui float
  test = (h0);
  // CHECK: uitofp i32
  // NOHALF: [[F32TOF16:call i16 @llvm.convert.to.fp16.f32]]
  // HALF: [[F32TOF16:fptrunc float]]
  h0 = (test);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  test = (!h1);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // NOHALF: [[F32TOF16]]
  // HALF: [[F32TOF16]]
  h1 = -h1;
  // CHECK: [[F16TOF32]]
  // CHECK: [[F32TOF16]]
  h1 = +h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h1++;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  ++h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  --h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h1--;

  // Check binary ops with various operands
  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  h1 = h0 * h2;
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F32TOF16]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  h1 = h0 * (__fp16) -2.0f;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  h1 = h0 * f2;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  h1 = f0 * h2;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  h1 = (h0 / h2);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  h1 = (h0 / (__fp16) -2.0f);
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  h1 = (h0 / f2);
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  h1 = (f0 / h2);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h1 = (h2 + h0);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h1 = ((__fp16)-2.0 + h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h1 = (h2 + f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h1 = (f2 + h0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  h1 = (h2 - h0);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  h1 = ((__fp16)-2.0f - h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  h1 = (h2 - f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  h1 = (f2 - h0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt
  test = (h2 < h0);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fcmp olt
  test = (h2 < (__fp16)42.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt
  test = (h2 < f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt
  test = (f2 < h0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt
  test = (h0 > h2);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fcmp ogt
  test = ((__fp16)42.0 > h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt
  test = (h0 > f2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt
  test = (f0 > h2);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole
  test = (h2 <= h0);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fcmp ole
  test = (h2 <= (__fp16)42.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole
  test = (h2 <= f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole
  test = (f2 <= h0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge
  test = (h0 >= h2);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fcmp oge
  test = (h0 >= (__fp16)-2.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge
  test = (h0 >= f2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge
  test = (f0 >= h2);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq
  test = (h1 == h2);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fcmp oeq
  test = (h1 == (__fp16)1.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq
  test = (h1 == f1);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq
  test = (f1 == h1);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une
  test = (h1 != h2);
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fcmp une
  test = (h1 != (__fp16)1.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une
  test = (h1 != f1);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une
  test = (f1 != h1);

  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une
  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: [[F32TOF16]]
  h1 = (h1 ? h2 : h0);
  // Check assignments (inc. compound)
  h0 = h1;
  // NOHALF: [[F32TOF16]]
  // HALF: store {{.*}} half 0xHC000
  h0 = (__fp16)-2.0f;
  // CHECK: [[F32TOF16]]
  h0 = f0;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  h0 += h1;
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fadd
  // CHECK: [[F32TOF16]]
  h0 += (__fp16)1.0f;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd
  // CHECK: [[F32TOF16]]
  h0 += f2;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fsub
  // CHECK: [[F32TOF16]]
  h0 -= h1;
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fsub
  // CHECK: [[F32TOF16]]
  h0 -= (__fp16)1.0;
  // CHECK: [[F16TOF32]]
  // CHECK: fsub
  // CHECK: [[F32TOF16]]
  h0 -= f2;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fmul
  // CHECK: [[F32TOF16]]
  h0 *= h1;
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fmul
  // CHECK: [[F32TOF16]]
  h0 *= (__fp16)1.0;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul
  // CHECK: [[F32TOF16]]
  h0 *= f2;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv
  // CHECK: [[F32TOF16]]
  h0 /= h1;
  // CHECK: [[F16TOF32]]
  // NOHALF: [[F16TOF32]]
  // CHECK: fdiv
  // CHECK: [[F32TOF16]]
  h0 /= (__fp16)1.0;
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv
  // CHECK: [[F32TOF16]]
  h0 /= f2;

  // Check conversions to/from double
  // NOHALF: call i16 @llvm.convert.to.fp16.f64(
  // HALF: fptrunc double {{.*}} to half
  h0 = d0;

  // CHECK: [[MID:%.*]] = fptrunc double {{%.*}} to float
  // NOHALF: call i16 @llvm.convert.to.fp16.f32(float [[MID]])
  // HALF: fptrunc float [[MID]] to half
  h0 = (float)d0;

  // NOHALF: call double @llvm.convert.from.fp16.f64(
  // HALF: fpext half {{.*}} to double
  d0 = h0;

  // NOHALF: [[MID:%.*]] = call float @llvm.convert.from.fp16.f32(
  // HALF: [[MID:%.*]] = fpext half {{.*}} to float
  // CHECK: fpext float [[MID]] to double
  d0 = (float)h0;
}
