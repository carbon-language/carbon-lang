// RUN: %clangxx -fsanitize=float-cast-overflow %s -o %t
// RUN: %run %t _
// RUN: %run %t 0 2>&1 | FileCheck %s --check-prefix=CHECK-0
// RUN: %run %t 1 2>&1 | FileCheck %s --check-prefix=CHECK-1
// RUN: %run %t 2 2>&1 | FileCheck %s --check-prefix=CHECK-2
// RUN: %run %t 3 2>&1 | FileCheck %s --check-prefix=CHECK-3
// RUN: %run %t 4 2>&1 | FileCheck %s --check-prefix=CHECK-4
// RUN: %run %t 5 2>&1 | FileCheck %s --check-prefix=CHECK-5
// RUN: %run %t 6 2>&1 | FileCheck %s --check-prefix=CHECK-6
// FIXME: %run %t 7 2>&1 | FileCheck %s --check-prefix=CHECK-7
// RUN: %run %t 8 2>&1 | FileCheck %s --check-prefix=CHECK-8

// This test assumes float and double are IEEE-754 single- and double-precision.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

float Inf;
float NaN;

int main(int argc, char **argv) {
  float MaxFloatRepresentableAsInt = 0x7fffff80;
  (int)MaxFloatRepresentableAsInt; // ok
  (int)-MaxFloatRepresentableAsInt; // ok

  float MinFloatRepresentableAsInt = -0x7fffffff - 1;
  (int)MinFloatRepresentableAsInt; // ok

  float MaxFloatRepresentableAsUInt = 0xffffff00u;
  (unsigned int)MaxFloatRepresentableAsUInt; // ok

#ifdef __SIZEOF_INT128__
  unsigned __int128 FloatMaxAsUInt128 = -((unsigned __int128)1 << 104);
  (void)(float)FloatMaxAsUInt128; // ok
#endif

  float NearlyMinusOne = -0.99999;
  unsigned Zero = NearlyMinusOne; // ok

  // Build a '+Inf'.
  char InfVal[] = { 0x00, 0x00, 0x80, 0x7f };
  float Inf;
  memcpy(&Inf, InfVal, 4);

  // Build a 'NaN'.
  char NaNVal[] = { 0x01, 0x00, 0x80, 0x7f };
  float NaN;
  memcpy(&NaN, NaNVal, 4);

  double DblInf = (double)Inf; // ok

  switch (argv[1][0]) {
    // FIXME: Produce a source location for these checks and test for it here.

    // Floating point -> integer overflow.
  case '0':
    // Note that values between 0x7ffffe00 and 0x80000000 may or may not
    // successfully round-trip, depending on the rounding mode.
    // CHECK-0: runtime error: value 2.14748{{.*}} is outside the range of representable values of type 'int'
    return MaxFloatRepresentableAsInt + 0x80;
  case '1':
    // CHECK-1: runtime error: value -2.14748{{.*}} is outside the range of representable values of type 'int'
    return MinFloatRepresentableAsInt - 0x100;
  case '2':
    // CHECK-2: runtime error: value -1 is outside the range of representable values of type 'unsigned int'
    return (unsigned)-1.0;
  case '3':
    // CHECK-3: runtime error: value 4.2949{{.*}} is outside the range of representable values of type 'unsigned int'
    return (unsigned)(MaxFloatRepresentableAsUInt + 0x100);

  case '4':
    // CHECK-4: runtime error: value {{.*}} is outside the range of representable values of type 'int'
    return Inf;
  case '5':
    // CHECK-5: runtime error: value {{.*}} is outside the range of representable values of type 'int'
    return NaN;

    // Integer -> floating point overflow.
  case '6':
    // CHECK-6: {{runtime error: value 0xffffff00000000000000000000000001 is outside the range of representable values of type 'float'|__int128 not supported}}
#ifdef __SIZEOF_INT128__
    return (float)(FloatMaxAsUInt128 + 1);
#else
    puts("__int128 not supported");
    return 0;
#endif
  // FIXME: The backend cannot lower __fp16 operations on x86 yet.
  //case '7':
  //  (__fp16)65504; // ok
  //  // CHECK-7: runtime error: value 65505 is outside the range of representable values of type '__fp16'
  //  return (__fp16)65505;

    // Floating point -> floating point overflow.
  case '8':
    // CHECK-8: runtime error: value 1e+39 is outside the range of representable values of type 'float'
    return (float)1e39;
  }
}
