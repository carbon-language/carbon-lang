// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -S -o - -emit-llvm %s | FileCheck %s -check-prefix=CHECK-NOERRNO
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -o - -emit-llvm -fmath-errno %s | FileCheck %s -check-prefix=CHECK-ERRNO
// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -S -o - -emit-llvm -x c++ %s | FileCheck %s -check-prefix=CHECK-NOERRNO
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -o - -emit-llvm -x c++ -fmath-errno %s | FileCheck %s -check-prefix=CHECK-ERRNO

// Prototypes.
#ifdef __cplusplus
extern "C" {
#endif
double acos(double);
long double acosl(long double);
float acosf(float);
double asin(double);
long double asinl(long double);
float asinf(float);
double atan(double);
long double atanl(long double);
float atanf(float);
double atan2(double, double);
long double atan2l(long double, long double);
float atan2f(float, float);
double ceil(double);
long double ceill(long double);
float ceilf(float);
double copysign(double, double);
long double copysignl(long double, long double);
float copysignf(float, float);
double cos(double);
long double cosl(long double);
float cosf(float);
double exp(double);
long double expl(long double);
float expf(float);
double exp2(double);
long double exp2l(long double);
float exp2f(float);
double fabs(double);
long double fabsl(long double);
float fabsf(float);
double floor(double);
long double floorl(long double);
float floorf(float);
double fma(double, double, double);
long double fmal(long double, long double, long double);
float fmaf(float, float, float);
double fmax(double, double);
long double fmaxl(long double, long double);
float fmaxf(float, float);
double fmin(double, double);
long double fminl(long double, long double);
float fminf(float, float);
double log(double);
long double logl(long double);
float logf(float);
double log2(double);
long double log2l(long double);
float log2f(float);
long lrint(double);
long lrintl(long double);
long lrintf(float);
double nearbyint(double);
long double nearbyintl(long double);
float nearbyintf(float);
double pow(double, double);
long double powl(long double, long double);
float powf(float, float);
double rint(double);
long double rintl(long double);
float rintf(float);
double round(double);
long double roundl(long double);
float roundf(float);
double sin(double);
long double sinl(long double);
float sinf(float);
double sqrt(double);
long double sqrtl(long double);
float sqrtf(float);
double tan(double);
long double tanl(long double);
float tanf(float);
double trunc(double);
long double truncl(long double);
float truncf(float);
#ifdef __cplusplus
}
#endif

// Force emission of the declare statements.
#define F(x) ((void*)x)
void *use[] = { F(acos), F(acosl), F(acosf), F(asin), F(asinl), F(asinf),
                F(atan), F(atanl), F(atanf), F(atan2), F(atan2l), F(atan2f),
                F(ceil), F(ceill), F(ceilf), F(copysign), F(copysignl),
                F(copysignf), F(cos), F(cosl), F(cosf), F(exp), F(expl),
                F(expf), F(exp2), F(exp2l), F(exp2f), F(fabs), F(fabsl),
                F(fabsf), F(floor), F(floorl), F(floorf), F(fma), F(fmal),
                F(fmaf), F(fmax), F(fmaxl), F(fmaxf), F(fmin), F(fminl),
                F(fminf), F(log), F(logl), F(logf), F(log2), F(log2l),
                F(log2f), F(lrint), F(lrintl), F(lrintf), F(nearbyint),
                F(nearbyintl), F(nearbyintf), F(pow), F(powl), F(powf),
                F(rint), F(rintl), F(rintf), F(round), F(roundl), F(roundf),
                F(sin), F(sinl), F(sinf), F(sqrt), F(sqrtl), F(sqrtf), F(tan),
                F(tanl), F(tanf), F(trunc), F(truncl), F(truncf) };

// CHECK-NOERRNO: declare double @acos(double) [[NUW:#[0-9]+]]
// CHECK-NOERRNO: declare x86_fp80 @acosl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @acosf(float) [[NUW]]
// CHECK-NOERRNO: declare double @asin(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @asinl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @asinf(float) [[NUW]]
// CHECK-NOERRNO: declare double @atan(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @atanl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @atanf(float) [[NUW]]
// CHECK-NOERRNO: declare double @atan2(double, double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @atan2f(float, float) [[NUW]]
// CHECK-NOERRNO: declare double @ceil(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @ceill(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @ceilf(float) [[NUW]]
// CHECK-NOERRNO: declare double @copysign(double, double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @copysignf(float, float) [[NUW]]
// CHECK-NOERRNO: declare double @cos(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @cosl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @cosf(float) [[NUW]]
// CHECK-NOERRNO: declare double @exp(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @expl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @expf(float) [[NUW]]
// CHECK-NOERRNO: declare double @exp2(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @exp2l(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @exp2f(float) [[NUW]]
// CHECK-NOERRNO: declare double @fabs(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @fabsl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @fabsf(float) [[NUW]]
// CHECK-NOERRNO: declare double @floor(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @floorl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @floorf(float) [[NUW]]
// CHECK-NOERRNO: declare double @fma(double, double, double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @fmaf(float, float, float) [[NUW]]
// CHECK-NOERRNO: declare double @fmax(double, double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @fmaxf(float, float) [[NUW]]
// CHECK-NOERRNO: declare double @fmin(double, double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @fminf(float, float) [[NUW]]
// CHECK-NOERRNO: declare double @log(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @logl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @logf(float) [[NUW]]
// CHECK-NOERRNO: declare double @log2(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @log2l(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @log2f(float) [[NUW]]
// CHECK-NOERRNO: declare {{i..}} @lrint(double) [[NUW]]
// CHECK-NOERRNO: declare {{i..}} @lrintl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare {{i..}} @lrintf(float) [[NUW]]
// CHECK-NOERRNO: declare double @nearbyint(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @nearbyintl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @nearbyintf(float) [[NUW]]
// CHECK-NOERRNO: declare double @pow(double, double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @powf(float, float) [[NUW]]
// CHECK-NOERRNO: declare double @rint(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @rintl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @rintf(float) [[NUW]]
// CHECK-NOERRNO: declare double @round(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @roundl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @roundf(float) [[NUW]]
// CHECK-NOERRNO: declare double @sin(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @sinl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @sinf(float) [[NUW]]
// CHECK-NOERRNO: declare double @sqrt(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @sqrtl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @sqrtf(float) [[NUW]]
// CHECK-NOERRNO: declare double @tan(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @tanl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @tanf(float) [[NUW]]
// CHECK-NOERRNO: declare double @trunc(double) [[NUW]]
// CHECK-NOERRNO: declare x86_fp80 @truncl(x86_fp80) [[NUW]]
// CHECK-NOERRNO: declare float @truncf(float) [[NUW]]

// CHECK-ERRNO: declare double @ceil(double) [[NUW:#[0-9]+]]
// CHECK-ERRNO: declare x86_fp80 @ceill(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @ceilf(float) [[NUW]]
// CHECK-ERRNO: declare double @copysign(double, double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @copysignf(float, float) [[NUW]]
// CHECK-ERRNO: declare double @fabs(double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @fabsl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @fabsf(float) [[NUW]]
// CHECK-ERRNO: declare double @floor(double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @floorl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @floorf(float) [[NUW]]
// CHECK-ERRNO: declare double @fmax(double, double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @fmaxf(float, float) [[NUW]]
// CHECK-ERRNO: declare double @fmin(double, double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @fminf(float, float) [[NUW]]
// CHECK-ERRNO: declare {{i..}} @lrint(double) [[NUW]]
// CHECK-ERRNO: declare {{i..}} @lrintl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare {{i..}} @lrintf(float) [[NUW]]
// CHECK-ERRNO: declare double @nearbyint(double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @nearbyintl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @nearbyintf(float) [[NUW]]
// CHECK-ERRNO: declare double @rint(double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @rintl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @rintf(float) [[NUW]]
// CHECK-ERRNO: declare double @round(double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @roundl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @roundf(float) [[NUW]]
// CHECK-ERRNO: declare double @trunc(double) [[NUW]]
// CHECK-ERRNO: declare x86_fp80 @truncl(x86_fp80) [[NUW]]
// CHECK-ERRNO: declare float @truncf(float) [[NUW]]

// CHECK-NOERRNO: attributes [[NUW]] = { nounwind readnone{{.*}} }

// CHECK-ERRNO: attributes [[NUW]] = { nounwind readnone{{.*}} }
