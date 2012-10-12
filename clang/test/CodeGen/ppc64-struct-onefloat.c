// REQUIRES: ppc64-registered-target
// RUN: %clang_cc1 -O0 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef struct s1 { float f; } Sf;
typedef struct s2 { double d; } Sd;
typedef struct s3 { long double ld; } Sld;
typedef struct s4 { Sf fs; } SSf;
typedef struct s5 { Sd ds; } SSd;
typedef struct s6 { Sld lds; } SSld;

void bar(Sf a, Sd b, Sld c, SSf d, SSd e, SSld f) {}

// CHECK: define void @bar
// CHECK:  %a = alloca %struct.s1, align 4
// CHECK:  %b = alloca %struct.s2, align 8
// CHECK:  %c = alloca %struct.s3, align 16
// CHECK:  %d = alloca %struct.s4, align 4
// CHECK:  %e = alloca %struct.s5, align 8
// CHECK:  %f = alloca %struct.s6, align 16
// CHECK:  %coerce.dive = getelementptr %struct.s1* %a, i32 0, i32 0
// CHECK:  store float %a.coerce, float* %coerce.dive, align 1
// CHECK:  %coerce.dive1 = getelementptr %struct.s2* %b, i32 0, i32 0
// CHECK:  store double %b.coerce, double* %coerce.dive1, align 1
// CHECK:  %coerce.dive2 = getelementptr %struct.s3* %c, i32 0, i32 0
// CHECK:  store ppc_fp128 %c.coerce, ppc_fp128* %coerce.dive2, align 1
// CHECK:  %coerce.dive3 = getelementptr %struct.s4* %d, i32 0, i32 0
// CHECK:  %coerce.dive4 = getelementptr %struct.s1* %coerce.dive3, i32 0, i32 0
// CHECK:  store float %d.coerce, float* %coerce.dive4, align 1
// CHECK:  %coerce.dive5 = getelementptr %struct.s5* %e, i32 0, i32 0
// CHECK:  %coerce.dive6 = getelementptr %struct.s2* %coerce.dive5, i32 0, i32 0
// CHECK:  store double %e.coerce, double* %coerce.dive6, align 1
// CHECK:  %coerce.dive7 = getelementptr %struct.s6* %f, i32 0, i32 0
// CHECK:  %coerce.dive8 = getelementptr %struct.s3* %coerce.dive7, i32 0, i32 0
// CHECK:  store ppc_fp128 %f.coerce, ppc_fp128* %coerce.dive8, align 1
// CHECK:  ret void

void foo(void) 
{
  Sf p1 = { 22.63f };
  Sd p2 = { 19.47 };
  Sld p3 = { -155.1l };
  SSf p4 = { { 22.63f } };
  SSd p5 = { { 19.47 } };
  SSld p6 = { { -155.1l } };
  bar(p1, p2, p3, p4, p5, p6);
}

// CHECK: define void @foo
// CHECK:  %coerce.dive = getelementptr %struct.s1* %p1, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load float* %coerce.dive, align 1
// CHECK:  %coerce.dive1 = getelementptr %struct.s2* %p2, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load double* %coerce.dive1, align 1
// CHECK:  %coerce.dive2 = getelementptr %struct.s3* %p3, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load ppc_fp128* %coerce.dive2, align 1
// CHECK:  %coerce.dive3 = getelementptr %struct.s4* %p4, i32 0, i32 0
// CHECK:  %coerce.dive4 = getelementptr %struct.s1* %coerce.dive3, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load float* %coerce.dive4, align 1
// CHECK:  %coerce.dive5 = getelementptr %struct.s5* %p5, i32 0, i32 0
// CHECK:  %coerce.dive6 = getelementptr %struct.s2* %coerce.dive5, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load double* %coerce.dive6, align 1
// CHECK:  %coerce.dive7 = getelementptr %struct.s6* %p6, i32 0, i32 0
// CHECK:  %coerce.dive8 = getelementptr %struct.s3* %coerce.dive7, i32 0, i32 0
// CHECK:  %{{[0-9]+}} = load ppc_fp128* %coerce.dive8, align 1
// CHECK:  call void @bar(float inreg %{{[0-9]+}}, double inreg %{{[0-9]+}}, ppc_fp128 inreg %{{[0-9]+}}, float inreg %{{[0-9]+}}, double inreg %{{[0-9]+}}, ppc_fp128 inreg %{{[0-9]+}})
// CHECK:  ret void
