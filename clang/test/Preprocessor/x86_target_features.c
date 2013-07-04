// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4 -x c -E -dM -o - %s | FileCheck --check-prefix=SSE4 %s

// SSE4: #define __SSE2_MATH__ 1
// SSE4: #define __SSE2__ 1
// SSE4: #define __SSE3__ 1
// SSE4: #define __SSE4_1__ 1
// SSE4: #define __SSE4_2__ 1
// SSE4: #define __SSE_MATH__ 1
// SSE4: #define __SSE__ 1
// SSE4: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4 -mno-sse2 -x c -E -dM -o - %s | FileCheck --check-prefix=SSE %s

// SSE-NOT: #define __SSE2_MATH__ 1
// SSE-NOT: #define __SSE2__ 1
// SSE-NOT: #define __SSE3__ 1
// SSE-NOT: #define __SSE4_1__ 1
// SSE-NOT: #define __SSE4_2__ 1
// SSE: #define __SSE_MATH__ 1
// SSE: #define __SSE__ 1
// SSE-NOT: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium-m -x c -E -dM -o - %s | FileCheck --check-prefix=SSE2 %s

// SSE2: #define __SSE2_MATH__ 1
// SSE2: #define __SSE2__ 1
// SSE2-NOT: #define __SSE3__ 1
// SSE2-NOT: #define __SSE4_1__ 1
// SSE2-NOT: #define __SSE4_2__ 1
// SSE2: #define __SSE_MATH__ 1
// SSE2: #define __SSE__ 1
// SSE2-NOT: #define __SSSE3__ 1
